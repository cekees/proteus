#ifndef PROTEUS_PYARRAY_H
#define PROTEUS_PYARRAY_H

// proteus::pyarray<T> -- a non-owning view of a numpy array, built on
// pybind11 and the numpy C-API only.
//
// This replaces xt::pyarray<T> (xtensor-python), which proteus used purely
// as an array-view type: of xtensor's very large API, the extensions used
// data(), size(), shape(i), operator[] and operator(), and nothing else.
// Carrying xtl/xtensor/xtensor-python for that has cost more in dependency
// breakage than the type is worth (see ROADMAP_1.9_2.0.md Phase 2 item 1).
//
// Semantics deliberately kept identical to xt::pyarray for the operations
// proteus uses:
//
//   * The view does not copy. It holds a reference to the numpy array, so
//     writes through it are visible in Python -- which every kernel here
//     relies on for its output arrays.
//   * operator()(i0, i1, ...) uses xtensor's index alignment rule: with m
//     indices and rank n, if m == n the indices are used as given; if m < n
//     the indices align with the *trailing* n - m dimensions; if m > n the
//     leading m - n indices are ignored.
//   * operator[](i) is operator()(i). On a C-contiguous array the trailing
//     stride is 1, so for rank >= 1 this is the flat index i -- which is
//     what the extensions assume, since they freely mix arr[i] with
//     arr.data()[i] on the same array.
//
// One behaviour is deliberately NOT preserved. xt::pyarray's caster built its
// view with PyArray_FromAny(..., NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST),
// so it accepted *anything* array-like -- a tuple, a list, a float32 array
// where double was wanted, a strided slice -- by silently making a converted
// copy. For an input array that is merely wasteful; for an output array it is
// a correctness trap, because the kernel then writes into the copy and Python
// never sees the result. (It had also been quietly papering over eleven
// `argsDict[...] = arr,` typos on the Python side -- a trailing comma makes
// the value a 1-tuple -- all of which are now fixed.)
//
// So this view converts nothing: the array must already be of T's dtype and
// C-contiguous, or the call fails. A dtype mismatch fails the pybind11 load
// quietly, letting overload resolution move on; a right-dtype array with the
// wrong layout throws, since nothing else could have been meant by it.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace proteus
{
    // numpy allows rank up to 64; proteus's arrays are rank <= 5. A fixed
    // bound keeps shape/strides inline, so copying a view is just a refcount
    // bump and operator() needs no pointer chase.
    static constexpr std::size_t pyarray_max_dimension = 10;

    /***********
     * pyarray *
     ***********/

    template <class T>
    class pyarray
    {
    public:

        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using iterator = T*;
        using const_iterator = const T*;

        // Minimal stand-in for xtensor's shape container: enough for
        // shape().size() / shape()[i] / iteration.
        class shape_view
        {
        public:

            shape_view(const size_type* p, size_type n) noexcept
                : m_p(p), m_n(n)
            {
            }

            size_type size() const noexcept { return m_n; }
            size_type operator[](size_type i) const noexcept { return m_p[i]; }
            const size_type* begin() const noexcept { return m_p; }
            const size_type* end() const noexcept { return m_p + m_n; }

        private:

            const size_type* m_p;
            size_type m_n;
        };

        pyarray() = default;

        // Adopt an existing numpy array. Throws if it is not a C-contiguous
        // array of T.
        explicit pyarray(pybind11::object obj);

        // Allocate a new (uninitialized) numpy array of the given shape.
        static pyarray from_shape(std::initializer_list<size_type> shape);

        pointer data() noexcept { return m_data; }
        const_pointer data() const noexcept { return m_data; }

        size_type size() const noexcept { return m_size; }
        size_type dimension() const noexcept { return m_dimension; }

        shape_view shape() const noexcept
        {
            return shape_view(m_shape.data(), m_dimension);
        }

        size_type shape(size_type i) const noexcept { return m_shape[i]; }

        iterator begin() noexcept { return m_data; }
        iterator end() noexcept { return m_data + m_size; }
        const_iterator begin() const noexcept { return m_data; }
        const_iterator end() const noexcept { return m_data + m_size; }
        const_iterator cbegin() const noexcept { return m_data; }
        const_iterator cend() const noexcept { return m_data + m_size; }

        template <class... Args>
        reference operator()(Args... args) noexcept
        {
            return m_data[data_offset(args...)];
        }

        template <class... Args>
        const_reference operator()(Args... args) const noexcept
        {
            return m_data[data_offset(args...)];
        }

        reference operator[](size_type i) noexcept { return (*this)(i); }
        const_reference operator[](size_type i) const noexcept { return (*this)(i); }

        // Bounds-checked access, matching xt::pyarray::at. Same index
        // alignment as operator(), plus the two checks xtensor makes:
        // more indices than the rank is an error here (operator() silently
        // drops the leading ones), and each index is range-checked against
        // its axis. An axis of length 1 is exempt, as it is in xtensor,
        // where it exists to let broadcasting through.
        //
        // Note the alignment applies to the check too: at(i) on a rank-2
        // array checks i against the *last* dimension, not the first. That
        // is xtensor's behaviour, surprising as it is; every call site in
        // proteus is on a rank-1 array, where it is simply the flat bound.
        template <class... Args>
        reference at(Args... args)
        {
            check_access(args...);
            return (*this)(args...);
        }

        template <class... Args>
        const_reference at(Args... args) const
        {
            check_access(args...);
            return (*this)(args...);
        }

        // The underlying numpy array, for handing back to Python.
        const pybind11::object& pyobject() const noexcept { return m_obj; }

    private:

        template <class... Args>
        size_type data_offset(Args... args) const noexcept;

        template <class... Args>
        void check_access(Args... args) const;

        pybind11::object m_obj;
        T* m_data = nullptr;
        size_type m_size = 0;
        size_type m_dimension = 0;
        std::array<size_type, pyarray_max_dimension> m_shape = {};
        std::array<size_type, pyarray_max_dimension> m_strides = {};
    };

    /**********************
     * dtype/layout check *
     **********************/

    namespace detail
    {
        // True if src is a numpy array whose dtype is exactly T's. Does not
        // look at the layout -- that is checked (and reported) separately, so
        // that an array of the wrong dtype can fall through to another
        // pybind11 overload while a badly laid out array of the *right* dtype
        // raises instead of being silently mishandled.
        template <class T>
        inline bool is_array_of(pybind11::handle src)
        {
            const auto& api = pybind11::detail::npy_api::get();
            return src
                && api.PyArray_Check_(src.ptr())
                && api.PyArray_EquivTypes_(pybind11::detail::array_proxy(src.ptr())->descr,
                                           pybind11::dtype::of<T>().ptr());
        }
    }

    /**************************
     * pyarray implementation *
     **************************/

    template <class T>
    inline pyarray<T>::pyarray(pybind11::object obj)
        : m_obj(std::move(obj))
    {
        namespace py = pybind11;

        if (!detail::is_array_of<T>(m_obj))
        {
            throw py::type_error("proteus::pyarray: expected a numpy array of dtype "
                                 + std::string(py::str(py::dtype::of<T>())));
        }
        if (!py::detail::check_flags(m_obj.ptr(), py::array::c_style))
        {
            throw py::type_error("proteus::pyarray: expected a C-contiguous numpy array; "
                                 "proteus's kernels index the array's data() buffer "
                                 "directly and cannot honour arbitrary strides");
        }

        auto arr = py::reinterpret_borrow<py::array>(m_obj);
        m_dimension = static_cast<size_type>(arr.ndim());
        if (m_dimension > pyarray_max_dimension)
        {
            throw py::type_error("proteus::pyarray: array rank "
                                 + std::to_string(m_dimension)
                                 + " exceeds the supported maximum of "
                                 + std::to_string(pyarray_max_dimension));
        }
        m_size = static_cast<size_type>(arr.size());

        for (size_type i = 0; i < m_dimension; ++i)
        {
            m_shape[i] = static_cast<size_type>(arr.shape(static_cast<py::ssize_t>(i)));
        }
        // Recomputed from the shape rather than read from numpy: the array is
        // known contiguous here, and numpy reports arbitrary strides for
        // length-0 and length-1 dimensions.
        size_type stride = 1;
        for (size_type i = m_dimension; i-- > 0;)
        {
            m_strides[i] = stride;
            stride *= m_shape[i];
        }

        // const_cast: proteus's kernels write their results into the arrays
        // handed in from Python, which is the whole point of the view. numpy's
        // own writeable flag is not consulted, matching xt::pyarray.
        m_data = static_cast<T*>(const_cast<void*>(arr.data()));
    }

    template <class T>
    inline pyarray<T> pyarray<T>::from_shape(std::initializer_list<size_type> shape)
    {
        std::vector<pybind11::ssize_t> s;
        s.reserve(shape.size());
        for (size_type n : shape)
        {
            s.push_back(static_cast<pybind11::ssize_t>(n));
        }
        pybind11::array_t<T, pybind11::array::c_style> a(s);
        return pyarray<T>(pybind11::reinterpret_borrow<pybind11::object>(a));
    }

    template <class T>
    template <class... Args>
    inline auto pyarray<T>::data_offset(Args... args) const noexcept -> size_type
    {
        constexpr size_type m = sizeof...(Args);
        if constexpr (m == 0)
        {
            return 0;
        }
        else
        {
            const std::array<difference_type, m> idx{{static_cast<difference_type>(args)...}};
            const size_type n = m_dimension;
            const size_type used = (m < n) ? m : n;
            const size_type first_index = (m > n) ? (m - n) : 0;
            const size_type first_stride = (m < n) ? (n - m) : 0;
            difference_type offset = 0;
            for (size_type t = 0; t < used; ++t)
            {
                offset += idx[first_index + t]
                        * static_cast<difference_type>(m_strides[first_stride + t]);
            }
            return static_cast<size_type>(offset);
        }
    }

    template <class T>
    template <class... Args>
    inline void pyarray<T>::check_access(Args... args) const
    {
        constexpr size_type m = sizeof...(Args);
        if (m > m_dimension)
        {
            throw std::out_of_range("proteus::pyarray: " + std::to_string(m)
                                    + " indices given for an array of rank "
                                    + std::to_string(m_dimension));
        }
        if constexpr (m > 0)
        {
            const std::array<difference_type, m> idx{{static_cast<difference_type>(args)...}};
            const size_type first_dim = m_dimension - m;
            for (size_type t = 0; t < m; ++t)
            {
                const size_type extent = m_shape[first_dim + t];
                if (extent != 1
                    && (idx[t] < 0 || static_cast<size_type>(idx[t]) >= extent))
                {
                    throw std::out_of_range("proteus::pyarray: index "
                                            + std::to_string(idx[t])
                                            + " is out of bounds for axis "
                                            + std::to_string(first_dim + t)
                                            + " with size " + std::to_string(extent));
                }
            }
        }
    }

    template <class T>
    inline std::ostream& operator<<(std::ostream& os, const pyarray<T>& a)
    {
        if (!a.pyobject())
        {
            return os << "<empty array>";
        }
        return os << std::string(pybind11::str(pybind11::repr(a.pyobject())));
    }

    /****************
     * import_numpy *
     ****************/

    // pybind11 resolves the numpy C-API lazily, on first use. Modules call
    // this at import time so that a broken/missing numpy is reported when the
    // extension is imported rather than in the middle of a solve -- the same
    // guarantee xt::import_numpy() gave.
    inline void import_numpy()
    {
        pybind11::detail::npy_api::get();
    }
}

/**************************
 * pybind11 type caster   *
 **************************/

namespace pybind11
{
    namespace detail
    {
        template <class T>
        struct type_caster<proteus::pyarray<T>>
        {
            PYBIND11_TYPE_CASTER(proteus::pyarray<T>,
                                 const_name("numpy.ndarray[")
                                     + npy_format_descriptor<T>::name
                                     + const_name("]"));

            bool load(handle src, bool)
            {
                // Reject a mismatched dtype quietly, so pybind11 can go on to
                // try other overloads. Anything else wrong with an array that
                // *is* of this dtype throws out of the pyarray constructor.
                if (!proteus::detail::is_array_of<T>(src))
                {
                    return false;
                }
                value = proteus::pyarray<T>(reinterpret_borrow<object>(src));
                return true;
            }

            static handle cast(const proteus::pyarray<T>& src, return_value_policy, handle)
            {
                if (!src.pyobject())
                {
                    return none().release();
                }
                return object(src.pyobject()).release();
            }
        };
    }
}

#endif
