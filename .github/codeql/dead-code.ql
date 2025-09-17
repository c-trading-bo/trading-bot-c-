/**
 * @name Dead Code Detection
 * @description Find methods, classes, and fields that are not used anywhere in the codebase
 * @kind problem
 * @problem.severity recommendation
 * @id csharp/dead-code
 * @tags maintainability
 *       dead-code
 */

import csharp

predicate isEntryPoint(Method m) {
  // Main methods
  m.getName() = "Main" and m.isStatic()
  or
  // Test methods (keep these even if unused)
  m.hasAttribute("Test") or m.hasAttribute("TestMethod") or m.hasAttribute("Fact")
  or
  // Controller actions (ASP.NET)
  m.getDeclaringType().getABaseType*().hasQualifiedName("Microsoft.AspNetCore.Mvc.Controller")
  or
  // SignalR Hub methods
  m.getDeclaringType().getABaseType*().hasQualifiedName("Microsoft.AspNetCore.SignalR.Hub")
  or
  // Orchestrator entry points
  m.getDeclaringType().getName().toLowerCase().matches("%orchestrator%") and
  (m.getName().matches("Start%") or m.getName().matches("Run%") or m.getName().matches("Execute%"))
  or
  // Background services
  m.getDeclaringType().getABaseType*().hasQualifiedName("Microsoft.Extensions.Hosting.BackgroundService")
  or
  // Public API methods that might be called externally
  m.isPublic() and m.getDeclaringType().isPublic() and 
  m.getDeclaringType().getNamespace().getName().matches("%.Api.%")
}

predicate isReachable(Method m) {
  isEntryPoint(m)
  or
  exists(Method caller | isReachable(caller) and caller.calls(m))
}

predicate isReachable(Field f) {
  exists(Method m | isReachable(m) and m.accesses(f))
}

predicate isReachable(Class c) {
  exists(Method m | m.getDeclaringType() = c and isReachable(m))
  or
  exists(Field f | f.getDeclaringType() = c and isReachable(f))
  or
  exists(Class other | isReachable(other) and other.getABaseType*() = c)
}

from Member member
where
  // Methods that are not reachable
  (member instanceof Method and not isReachable(member.(Method)) and 
   not member.(Method).isOverride() and not member.(Method).isVirtual() and
   not member.(Method).hasAttribute("JsonPropertyName") and
   not member.(Method).isGetter() and not member.(Method).isSetter())
  or
  // Private fields that are not used
  (member instanceof Field and not isReachable(member.(Field)) and 
   member.(Field).isPrivate() and not member.(Field).hasAttribute("JsonPropertyName"))
  or
  // Classes that are not used
  (member instanceof Class and not isReachable(member.(Class)) and
   not member.(Class).isPublic() and not member.(Class).hasAttribute("JsonObject"))
select member, "This " + member.getElementType() + " appears to be unused and could be removed."