.. contents::
    :local:
    :depth: 2


.. testsetup::

    # These are hidden from the docs, but these are necessary for `doctest`
    # since the `inspect` module doesn't play nicely with the execution
    # environment for `doctest`
    import torch

    original_script = torch.jit.script
    def script_wrapper(obj, *args, **kwargs):
        obj.__module__ = 'FakeMod'
        return original_script(obj, *args, **kwargs)

    torch.jit.script = script_wrapper

    original_trace = torch.jit.trace
    def trace_wrapper(obj, *args, **kwargs):
        obj.__module__ = 'FakeMod'
        return original_trace(obj, *args, **kwargs)

    torch.jit.trace = trace_wrapper

.. _language-reference:

TorchScript Language Reference
==============================

.. _type_annotation:


Type Annotation
~~~~~~~~~~~~~~~
Since TorchScript is a statically typed, programmers need to annotate types at *strategic points* of TorchScript codes so that every local variable or
instance data attribute has a static type, and every function and method has a statically typed signature.

When to annotate types
^^^^^^^^^^^^^^^^^^^^^^
In general, type annotations are only needed in places where static types cannot be automatically inferred, such as parameters or sometimes return types to
methods or functions. Types of local variables, data attributes are often automatically inferred from its assignment statements. Sometimes, an inferred type
may be too restrictive, e.g., ``x`` being inferred as ``NoneType`` through assignment ``x = None``, whereas ``x`` is actually used as an ``Optional``. In such
cases, type annotations may be needed to overwrite auto inference, e.g., x: ``Optional[int] = None``. Note that it is always safe to type annotate a local variable
or data attribute even if its type can be automatically inferred. But the annotated type must be congruent with TorchScript’s type checking.

When a parameter, local variable, or data attribute is not type annotated and its type cannot be automatically inferred, TorchScript assumes it to be a
default type of ``TensorType``.

Annotate function signature
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since parameter may not be automatically inferred from the body of the function (including both functions and methods), they need to be type annotated,
otherwise they assume the default type ``TensorType``.

TorchScript supports two styles for method and function signature type annotation:

* **Python3-style** annotates types directly on the signature. As such, it allows individual parameters be left unannotated
(whose type will be the default type of ``TensorType``) , or the return type be left unannotated (whose type will be automatically inferred).

::

    Python3Annotation := "def" Identifier [ "(" ParamAnnot* ")" ] [ReturnAnnot] ":"
                         FuncOrMethodBody
    ParamAnnot := Identifier [ ":" TSType ] ","
    ReturnAnnot := "->" TSType

Note that using Python3 style, the type of ``self`` is automatically inferred and should not be annotated.

* **Mypy style** annotates types as a comment right below the function/method declaration. In the My-Py style, since parameter names do not appear
in the annotation, all parameters have to be annotated.

::

    MyPyAnnotation := "# type:" "(" ParamAnnot* ")" [ ReturnAnnot ]
    ParamAnnot := TSType ","
    ReturnAnnot := "->" TSType

**Example 1**

In this example, ``a`` is not annotated and assumes the default type of ``TensorType``, ``b`` is annotated as type ``int``, and the return type is not
annotated and is automatically inferred as type ``TensorType``.

::

    import torch

    def f(a, b: int):
        return a+b

    m = torch.jit.script(f)
    print("TorchScript:", m(torch.ones([6]), 100))

**Example 2**

The following code snippet gives an example of using mypy style annotation. Note that parameters or return values must be annotated even if some of
them assume the default type.

::

    import torch

    def f(a, b):
        # type: (torch.Tensor, int) → torch.Tensor
        return a+b

    m = torch.jit.script(f)
    print("TorchScript:", m(torch.ones([6]), 100))

**Example 3**

This example caused a compilation error because it type annotates ``self``. The solution is to remove the type annotation and let its type be automatically
inferred.

::

    import torch

    @torch.jit.script
    class MyClass(object):
    # ERROR: do not annotate the type of self
    def __init__(self: MyClass, x: int):
        self.x = x

    def inc(self, val: int):
        self.x += val

Annotate variables and data attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general, types of data attributes (including class and instance data attributes) and local variables can be automatically inferred from assignment statements.
Sometimes, however, if a variable or attribute is associated with values of different types (e.g., as ``None`` or ``TensorType``), then they may need to be explicitly
type annotated as a *wider* type such as ``Optional[int]`` or ``Any``.

Local variables
"""""""""""""""
Local variables can be annotated according to Python3 typing module annotation rule, i.e.,

::

    LocalVarAnnotation := Identifier [":" TSType] "=" Expr

In general, types of local variables can be automatically inferred. In some cases, however, programmers may need to annotate a multi-type for local variables
that may be associated with different concrete types. Typical multi-types include ``Optional[T]`` and ``Any``.

**Example**

::

    import torch

    def f(a, setVal: bool):
        value: Optional[torch.Tensor] = None
        if setVal:
            value = a
        return value

    ones = torch.ones([6])
    m = torch.jit.script(f)
    print("TorchScript:", m(ones, True), m(ones, False))

Instance data attributes
""""""""""""""""""""""""
Instance data attributes can be annotated according to Python3 typing module annotation rules. Instance data attributes can be annotated (optionally) as final
via ``torch.jit.Final``.

::

    "class" ClassIdentifier ["(object)"] ":"
        InstanceAttrIdentifier ":" TSType

        "def __init__(self ("," ParameterAnnotation)* "):"
            "self." InstanceAttrIdentifier "=" Expr
            ...

where ``InstanceAttrIdentifier`` is the name of an instance attribute. For ``ModuleType`` classes, instance attributes may be declared as *final* via
``torch.jit.Final``:

::

    "class" ClassIdentifier "(torch.nn.Module):"
    InstanceAttrIdentifier ":" ["torch.jit.Final("] TSType [")"]
    ...

where ``InstanceAttrIdentifier`` is the name of an instance attribute and ``torch.jit.Final`` indicates that the attribute cannot be re-assigned outside
of ``__init__`` or overridden in subclasses.

**Example**

TODO: Add the example


Type Annotation APIs
^^^^^^^^^^^^^^^^^^^^

``torch.jit.annotate(T, expr)``
"""""""""""""""""""""""""""""""
This API annotates type ``T`` to an expression ``expr``. This is often used when the default type of an expression is not the type intended by the programmer.
For instance, an empty list (dictionary) has the default type of ``List[TensorType]`` (``Dict[TensorType, TensorType]``) but sometimes it may be used to initialize
a list of some other types.

**Example**

In this example, ``[]`` is declared as a list of integers via ``torch.jit.annotate`` (instead of assuming ``[]`` to be the default type of ``List[TensorType]``).

::

    import torch
    from typing import List

    def f(append: bool, val: int):
        l = torch.jit.annotate(List[int], [])
        if append:
            l.append(val)
            return l
        else:
            return None

    m = torch.jit.script(f)
    print("Eager:", f(True, 1), f(False, 1))
    print("TorchScript:", m(True, 1), m(False, 1))

TODO: Link to section about ``torch.jit.annotate``.


Appendix
^^^^^^^^

Unsupported Typing Constructs
"""""""""""""""""""""""""""""
TorchScript does not support all features and types of the Python3 `typing <https://docs.python.org/3/library/typing.html#module-typing>`_ module.
Any functionality from the typing `typing <https://docs.python.org/3/library/typing.html#module-typing>`_ module not explicitly specified in this
documentation is unsupported. The following table summarizes ``typing`` constructs that are either unsupported or supported with restrictions in TorchScript.

=============================  ================
 Item                           Description
-----------------------------  ----------------
``typing.Any``                  In development
``typing.NoReturn``             Not supported
``typing.Union``                In development
``typing.Callable``             Not supported
``typing.Literal``              Not supported
``typing.ClassVar``             Not supported
``typing.Final``                Supported for module attributes, class attribute, and annotations but not for functions
``typing.AnyStr``               Not supported
``typing.overload``             In development
Type aliases                    Not supported
Nominal typing                  In development
Structural typing               Not supported
NewType                         Not supported
Generics                        Not supported
=============================  ================
