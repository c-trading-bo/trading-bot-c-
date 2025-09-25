### Summary
What rules (IDs) were fixed and why.

### Proof
- [ ] `dotnet build -warnaserror` green
- [ ] `dotnet test` green
- [ ] Attached `tools/analyzers/current.sarif`
- [ ] Read and followed **docs/Analyzer-Fix-Guidebook.md v1.0**
- [ ] No posture edits (.editorconfig/props/csproj)
- [ ] No suppressions (#pragma / SuppressMessage / NoWarn)
- [ ] No new public setters on collections/domain state

### Notes
List 3–5 examples: Rule → file:line → brief "correct fix".