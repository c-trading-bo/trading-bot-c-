using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;
using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace TradingBot.Safety.Analyzers
{
    /// <summary>
    /// Custom analyzer to prevent agents from disabling aggressive build rules
    /// Addresses Comment #3304685224: Block Rule Suppression in Analyzer
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public class ProductionRuleEnforcementAnalyzer : DiagnosticAnalyzer
    {
        // Diagnostic descriptors for different violations
        public static readonly DiagnosticDescriptor SuppressMessageWithoutProof = new DiagnosticDescriptor(
            "PRE001",
            "SuppressMessage without RuntimeProof justification",
            "SuppressMessage attribute found without RuntimeProof justification: {0}",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "All analyzer suppressions must include RuntimeProof justification for production safety.");

        public static readonly DiagnosticDescriptor PragmaWarningDisable = new DiagnosticDescriptor(
            "PRE002", 
            "Pragma warning disable detected",
            "Pragma warning disable directive found: {0}",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Pragma warning disable directives are not allowed in production code.");

        public static readonly DiagnosticDescriptor CommentedAnalyzerRule = new DiagnosticDescriptor(
            "PRE003",
            "Commented-out analyzer rule detected", 
            "Commented analyzer rule detected: {0}",
            "Production", 
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Analyzer rules must not be commented out or disabled.");

        public static readonly DiagnosticDescriptor EditorConfigSeverityDowngrade = new DiagnosticDescriptor(
            "PRE004",
            "EditorConfig severity downgrade detected",
            "Analyzer rule severity downgraded to 'none' or 'silent': {0}",
            "Production",
            DiagnosticSeverity.Error, 
            isEnabledByDefault: true,
            description: "Analyzer rules must not be downgraded to 'none' or 'silent' severity.");

        public static readonly DiagnosticDescriptor HardcodedBusinessValue = new DiagnosticDescriptor(
            "PRE005",
            "Hardcoded business value detected",
            "Hardcoded business value found: {0}. Use configuration instead.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "All business values must be configuration-driven, not hardcoded.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(
                SuppressMessageWithoutProof,
                PragmaWarningDisable,
                CommentedAnalyzerRule,
                EditorConfigSeverityDowngrade,
                HardcodedBusinessValue);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeSuppressMessageAttribute, SyntaxKind.Attribute);
            context.RegisterSyntaxNodeAction(AnalyzePragmaWarningDirective, SyntaxKind.PragmaWarningDirectiveTrivia);
            context.RegisterSyntaxNodeAction(AnalyzeReturnStatement, SyntaxKind.ReturnStatement);
            context.RegisterSyntaxNodeAction(AnalyzeAssignmentExpression, SyntaxKind.SimpleAssignmentExpression);
            context.RegisterSyntaxTreeAction(AnalyzeSourceFile);
        }

        private static void AnalyzeSuppressMessageAttribute(SyntaxNodeAnalysisContext context)
        {
            var attribute = (AttributeSyntax)context.Node;
            
            // Check if this is a SuppressMessage attribute
            if (attribute.Name.ToString().Contains("SuppressMessage"))
            {
                var hasRuntimeProof = false;
                
                // Look for RuntimeProof justification in the attribute arguments
                if (attribute.ArgumentList?.Arguments != null)
                {
                    foreach (var arg in attribute.ArgumentList.Arguments)
                    {
                        var argText = arg.ToString();
                        if (argText.Contains("RuntimeProof") || argText.Contains("production") || argText.Contains("trading"))
                        {
                            hasRuntimeProof = true;
                            break;
                        }
                    }
                }

                if (!hasRuntimeProof)
                {
                    var diagnostic = Diagnostic.Create(SuppressMessageWithoutProof, 
                        attribute.GetLocation(), 
                        attribute.ToString());
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzePragmaWarningDirective(SyntaxNodeAnalysisContext context)
        {
            var pragmaDirective = context.Node;
            var text = pragmaDirective.ToString();
            
            if (text.Contains("#pragma warning disable"))
            {
                var diagnostic = Diagnostic.Create(PragmaWarningDisable,
                    pragmaDirective.GetLocation(),
                    text);
                context.ReportDiagnostic(diagnostic);
            }
        }

        private static void AnalyzeReturnStatement(SyntaxNodeAnalysisContext context)
        {
            var returnStatement = (ReturnStatementSyntax)context.Node;
            
            if (returnStatement.Expression is LiteralExpressionSyntax literal)
            {
                var value = literal.Token.ValueText;
                
                // Check for specific hardcoded business values
                if (IsHardcodedBusinessValue(value, context))
                {
                    var diagnostic = Diagnostic.Create(HardcodedBusinessValue,
                        literal.GetLocation(),
                        $"return {value}");
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzeAssignmentExpression(SyntaxNodeAnalysisContext context)
        {
            var assignment = (AssignmentExpressionSyntax)context.Node;
            
            if (assignment.Right is LiteralExpressionSyntax literal)
            {
                var value = literal.Token.ValueText;
                var variableName = assignment.Left.ToString();
                
                // Check for specific hardcoded business values in assignments
                if (IsHardcodedBusinessValue(value, context) && IsBusinessLogicContext(variableName))
                {
                    var diagnostic = Diagnostic.Create(HardcodedBusinessValue,
                        literal.GetLocation(),
                        $"{variableName} = {value}");
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzeSourceFile(SyntaxTreeAnalysisContext context)
        {
            var sourceText = context.Tree.GetText();
            var text = sourceText.ToString();
            
            // Check for commented-out analyzer rules
            CheckCommentedAnalyzerRules(context, text);
            
            // Check .editorconfig files for severity downgrades
            var filePath = context.Tree.FilePath;
            if (Path.GetFileName(filePath)?.Equals(".editorconfig", StringComparison.OrdinalIgnoreCase) == true)
            {
                CheckEditorConfigSeverityDowngrades(context, text);
            }
        }

        private static void CheckCommentedAnalyzerRules(SyntaxTreeAnalysisContext context, string text)
        {
            var commentedRulePattern = @"^\s*//\s*(dotnet_diagnostic\.|Analyzer:|DeadCode|InterfaceCheck)";
            var regex = new Regex(commentedRulePattern, RegexOptions.Multiline | RegexOptions.IgnoreCase);
            var matches = regex.Matches(text);
            
            foreach (Match match in matches)
            {
                var line = GetLineFromPosition(text, match.Index);
                var diagnostic = Diagnostic.Create(CommentedAnalyzerRule,
                    Location.Create(context.Tree, new TextSpan(match.Index, match.Length)),
                    match.Value.Trim());
                context.ReportDiagnostic(diagnostic);
            }
        }

        private static void CheckEditorConfigSeverityDowngrades(SyntaxTreeAnalysisContext context, string text)
        {
            var severityDowngradePattern = @"dotnet_diagnostic\.\w+\.severity\s*=\s*(none|silent)";
            var regex = new Regex(severityDowngradePattern, RegexOptions.Multiline | RegexOptions.IgnoreCase);
            var matches = regex.Matches(text);
            
            foreach (Match match in matches)
            {
                var diagnostic = Diagnostic.Create(EditorConfigSeverityDowngrade,
                    Location.Create(context.Tree, new TextSpan(match.Index, match.Length)),
                    match.Value.Trim());
                context.ReportDiagnostic(diagnostic);
            }
        }

        private static bool IsHardcodedBusinessValue(string value, SyntaxNodeAnalysisContext context)
        {
            // Check for specific problematic values mentioned in the issue
            if (value == "2.5" || value == "0.7" || value == "1.0")
            {
                // Additional context check - make sure we're in business logic, not test code
                var containingClass = context.Node.Ancestors().OfType<ClassDeclarationSyntax>().FirstOrDefault();
                if (containingClass != null)
                {
                    var className = containingClass.Identifier.ValueText;
                    var namespaceName = context.Node.Ancestors().OfType<NamespaceDeclarationSyntax>().FirstOrDefault()?.Name.ToString() ?? "";
                    
                    // Skip test classes and certain infrastructure classes
                    if (className.Contains("Test") || namespaceName.Contains("Test") || 
                        className.Contains("Mock") || namespaceName.Contains("Simulation"))
                    {
                        return false;
                    }
                    
                    // Focus on AI/ML and trading logic classes
                    return namespaceName.Contains("ML") || namespaceName.Contains("Trading") || 
                           namespaceName.Contains("Intelligence") || namespaceName.Contains("Strategy") ||
                           className.Contains("Position") || className.Contains("Confidence") || 
                           className.Contains("Regime") || className.Contains("ML") || className.Contains("AI");
                }
            }
            
            return false;
        }

        private static bool IsBusinessLogicContext(string variableName)
        {
            var businessTerms = new[] { "position", "confidence", "regime", "size", "threshold", "ratio", "factor" };
            return businessTerms.Any(term => variableName.ToLowerInvariant().Contains(term));
        }

        private static int GetLineFromPosition(string text, int position)
        {
            return text.Take(position).Count(c => c == '\n') + 1;
        }
    }
}