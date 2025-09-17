import csharp

from Method m
where not m.isAbstract()
  and not m.isExtern()
  and not m.getAReachableCallable()
select m, "This method is never called and can be removed or wired into the orchestrator."