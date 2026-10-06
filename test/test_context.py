import os, sys

def ContextObject():
    from collections import namedtuple
    globalSettings = {"nnx":11, "T":10.0, "g":9.8}
    MyContext = namedtuple("MyContext",list(globalSettings.keys()))
    return MyContext._make(list(globalSettings.values()))

def check_eq(context):
    assert context.nnx == 11
    assert context.T == 10.0
    assert context.g == 9.8

def test_set():
    from proteus import Context
    Context.set(ContextObject())
    check_eq(Context.context)

def test_setFromModule():
    import os
    from proteus import Context
    with open("context_module.py","w") as f:
        f.write("nnx=11; T=10.0; g=9.8\n")
    sys.path.append(os.getcwd())
    import context_module
    os.remove("context_module.py")
    Context.setFromModule(context_module)
    check_eq(Context.context)
    
def test_setMutableFromModule():
    import os
    from proteus import Context
    with open("context_module.py","w") as f:
        f.write("nnx=11; T=10.0; g=9.8\n")
    sys.path.append(os.getcwd())
    import context_module
    os.remove("context_module.py")
    Context.setFromModule(context_module, mutable=True)
    check_eq(Context.context)
    ct = Context.get()
    ct.T=11.0
    assert ct.T == 11.0

def test_get():
    from proteus import Context
    Context.set(ContextObject())
    ct = Context.get()
    check_eq(ct)
    try:
        ct.T=11.0
    except Exception as e:
        assert(type(e) is AttributeError)

def test_Options():
    import os
    from proteus import Context
    Context.contextOptionsString="nnx=11"
    with open("context_module.py","w") as f:
        f.write('from proteus import Context; opts=Context.Options([("nnx",12,"number of nodes")]); nnx=opts.nnx; T=10.0; g=9.8\n')
    sys.path.append(os.getcwd())
    import context_module
    os.remove("context_module.py")
    Context.setFromModule(context_module)
    check_eq(Context.context)

def test_splitContextOptions():
    from proteus.Context import _splitContextOptions
    # any run of whitespace (including newlines from an options file), commas, semicolons
    assert _splitContextOptions("a=1  b=2") == ["a=1", "b=2"]
    assert _splitContextOptions(" a=1 b=2 ") == ["a=1", "b=2"]
    assert _splitContextOptions("a=1,b=2;c=3") == ["a=1", "b=2", "c=3"]
    assert _splitContextOptions("a=1 ,; b=2\n\tc=3\n") == ["a=1", "b=2", "c=3"]
    assert _splitContextOptions("") == []
    # separators inside brackets and quotes are part of the value
    assert _splitContextOptions("L=[1.0, 2.0] x=(1,2);d={'k': 1}") == ["L=[1.0, 2.0]", "x=(1,2)", "d={'k': 1}"]
    assert _splitContextOptions("name='a b;c',other=\"d,e\"") == ["name='a b;c'", "other=\"d,e\""]

def test_Options_separators():
    from proteus import Context
    Context.contextOptionsString = "nnx=21,  T=5.0;\nL=[1.0, 2.0] name='a b' eq='x=y'"
    opts = Context.Options([("nnx", 11, ""), ("T", 10.0, ""), ("L", [0.0], ""), ("name", "", ""), ("eq", "", "")])
    Context.contextOptionsString = None
    assert opts.nnx == 21
    assert opts.T == 5.0
    assert opts.L == [1.0, 2.0]
    assert opts.name == "a b"
    assert opts.eq == "x=y"

def test_Options_malformed():
    import pytest
    from proteus import Context
    Context.contextOptionsString = "nnx=21 T"
    try:
        with pytest.raises(ValueError, match="name=value"):
            Context.Options([("nnx", 11, ""), ("T", 10.0, "")])
    finally:
        Context.contextOptionsString = None
