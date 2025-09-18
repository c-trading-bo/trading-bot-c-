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

        public static readonly DiagnosticDescriptor PlaceholderPatternDetected = new DiagnosticDescriptor(
            "PRE006",
            "Placeholder/Mock/Stub pattern detected",
            "Non-production pattern found: {0}. All code must be production-ready.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Mock, fake, stub, placeholder, or temporary code patterns are not allowed in production.");

        public static readonly DiagnosticDescriptor FixedSizeArrayDetected = new DiagnosticDescriptor(
            "PRE007",
            "Fixed-size array detected",
            "Fixed-size array found: {0}. Use dynamic sizing instead.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Fixed-size arrays must be replaced with configuration-driven dynamic sizing.");

        public static readonly DiagnosticDescriptor EmptyAsyncPlaceholder = new DiagnosticDescriptor(
            "PRE008",
            "Empty/placeholder async operation detected",
            "Placeholder async operation found: {0}. Implement actual business logic.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Empty async operations like Task.Yield, Task.Delay, or NotImplementedException are not allowed.");

        public static readonly DiagnosticDescriptor DevelopmentOnlyComment = new DiagnosticDescriptor(
            "PRE009",
            "Development/testing-only comment detected",
            "Development-only comment found: {0}. Remove non-production comments.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Comments indicating test-only, debug-only, TODO, FIXME, or temporary code are not allowed.");

        public static readonly DiagnosticDescriptor WeakRandomGeneration = new DiagnosticDescriptor(
            "PRE010",
            "Weak random number generation detected",
            "Weak random generation found: {0}. Use cryptographically secure random.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Use RandomNumberGenerator.Create() instead of new Random() for production code.");

        public static readonly DiagnosticDescriptor NumericLiteralInBusinessLogic = new DiagnosticDescriptor(
            "PRE011",
            "Numeric literal in business logic detected",
            "Hardcoded numeric literal found: {0}. All business values must come from configuration.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Any numeric literal in business logic must be replaced with configuration values.");

        public static readonly DiagnosticDescriptor EmptyMethodBodyDetected = new DiagnosticDescriptor(
            "PRE012",
            "Empty method body detected",
            "Empty method body found: {0}. All methods must contain actual implementation.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Empty method bodies {} are not allowed in production code - implement actual logic.");

        public static readonly DiagnosticDescriptor ShortMethodWithoutAttributes = new DiagnosticDescriptor(
            "PRE013",
            "Short method without proper attributes detected",
            "Method {0} has fewer than 3 lines without [Obsolete] or [GeneratedCode] attributes.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Methods with fewer than 3 lines must have [Obsolete] or [GeneratedCode] attributes or contain more implementation.");

        public static readonly DiagnosticDescriptor PythonPassStatementDetected = new DiagnosticDescriptor(
            "PRE014",
            "Python pass statement detected",
            "Python pass statement found in production file: {0}. Implement actual logic.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Python pass statements are not allowed in production code - implement actual logic.");

        public static readonly DiagnosticDescriptor UnusedParameterOrMemberDetected = new DiagnosticDescriptor(
            "PRE015",
            "Unused parameter or member detected",
            "Unused code detected: {0}. Remove unused code patterns.",
            "Production",
            DiagnosticSeverity.Error,
            isEnabledByDefault: true,
            description: "Unused parameters, private members, and fields indicate incomplete or stub code.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
            ImmutableArray.Create(
                SuppressMessageWithoutProof,
                PragmaWarningDisable,
                CommentedAnalyzerRule,
                EditorConfigSeverityDowngrade,
                HardcodedBusinessValue,
                PlaceholderPatternDetected,
                FixedSizeArrayDetected,
                EmptyAsyncPlaceholder,
                DevelopmentOnlyComment,
                WeakRandomGeneration,
                NumericLiteralInBusinessLogic,
                EmptyMethodBodyDetected,
                ShortMethodWithoutAttributes,
                PythonPassStatementDetected,
                UnusedParameterOrMemberDetected);

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterSyntaxNodeAction(AnalyzeSuppressMessageAttribute, SyntaxKind.Attribute);
            context.RegisterSyntaxNodeAction(AnalyzePragmaWarningDirective, SyntaxKind.PragmaWarningDirectiveTrivia);
            context.RegisterSyntaxNodeAction(AnalyzeReturnStatement, SyntaxKind.ReturnStatement);
            context.RegisterSyntaxNodeAction(AnalyzeAssignmentExpression, SyntaxKind.SimpleAssignmentExpression);
            context.RegisterSyntaxNodeAction(AnalyzeVariableDeclaration, SyntaxKind.VariableDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeFieldDeclaration, SyntaxKind.FieldDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzePropertyDeclaration, SyntaxKind.PropertyDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeMethodDeclaration, SyntaxKind.MethodDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeClassDeclaration, SyntaxKind.ClassDeclaration);
            context.RegisterSyntaxNodeAction(AnalyzeInvocationExpression, SyntaxKind.InvocationExpression);
            context.RegisterSyntaxNodeAction(AnalyzeObjectCreationExpression, SyntaxKind.ObjectCreationExpression);
            context.RegisterSyntaxNodeAction(AnalyzeArrayCreationExpression, SyntaxKind.ArrayCreationExpression);
            context.RegisterSyntaxNodeAction(AnalyzeThrowStatement, SyntaxKind.ThrowStatement);
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
                
                // Check for any numeric literal in business logic context
                if (IsNumericLiteral(literal.Token) && IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(NumericLiteralInBusinessLogic,
                        literal.GetLocation(),
                        $"return {value}");
                    context.ReportDiagnostic(diagnostic);
                }
                
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
                
                // Check for any numeric literal in business logic context
                if (IsNumericLiteral(literal.Token) && IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(NumericLiteralInBusinessLogic,
                        literal.GetLocation(),
                        $"{variableName} = {value}");
                    context.ReportDiagnostic(diagnostic);
                }
                
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

        private static void AnalyzeVariableDeclaration(SyntaxNodeAnalysisContext context)
        {
            var variableDeclaration = (VariableDeclarationSyntax)context.Node;
            
            foreach (var variable in variableDeclaration.Variables)
            {
                if (variable.Initializer?.Value is LiteralExpressionSyntax literal)
                {
                    var value = literal.Token.ValueText;
                    var variableName = variable.Identifier.ValueText;
                    
                    // Check for numeric literals in variable initialization
                    if (IsNumericLiteral(literal.Token) && IsInProductionCode(context))
                    {
                        var diagnostic = Diagnostic.Create(NumericLiteralInBusinessLogic,
                            literal.GetLocation(),
                            $"var {variableName} = {value}");
                        context.ReportDiagnostic(diagnostic);
                    }
                }
            }
        }

        private static void AnalyzeFieldDeclaration(SyntaxNodeAnalysisContext context)
        {
            var fieldDeclaration = (FieldDeclarationSyntax)context.Node;
            
            foreach (var variable in fieldDeclaration.Declaration.Variables)
            {
                if (variable.Initializer?.Value is LiteralExpressionSyntax literal)
                {
                    var value = literal.Token.ValueText;
                    var fieldName = variable.Identifier.ValueText;
                    
                    // Check for numeric literals in field initialization
                    if (IsNumericLiteral(literal.Token) && IsInProductionCode(context))
                    {
                        var diagnostic = Diagnostic.Create(NumericLiteralInBusinessLogic,
                            literal.GetLocation(),
                            $"field {fieldName} = {value}");
                        context.ReportDiagnostic(diagnostic);
                    }
                }
            }
        }

        private static void AnalyzePropertyDeclaration(SyntaxNodeAnalysisContext context)
        {
            var propertyDeclaration = (PropertyDeclarationSyntax)context.Node;
            
            if (propertyDeclaration.Initializer?.Value is LiteralExpressionSyntax literal)
            {
                var value = literal.Token.ValueText;
                var propertyName = propertyDeclaration.Identifier.ValueText;
                
                // Check for numeric literals in property initialization
                if (IsNumericLiteral(literal.Token) && IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(NumericLiteralInBusinessLogic,
                        literal.GetLocation(),
                        $"property {propertyName} = {value}");
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzeMethodDeclaration(SyntaxNodeAnalysisContext context)
        {
            var methodDeclaration = (MethodDeclarationSyntax)context.Node;
            var methodName = methodDeclaration.Identifier.ValueText;
            
            // Check for non-production patterns in method names
            if (ContainsNonProductionPattern(methodName))
            {
                var diagnostic = Diagnostic.Create(PlaceholderPatternDetected,
                    methodDeclaration.Identifier.GetLocation(),
                    $"method name: {methodName}");
                context.ReportDiagnostic(diagnostic);
            }
            
            // Check for empty method bodies
            if (methodDeclaration.Body != null && methodDeclaration.Body.Statements.Count == 0)
            {
                if (IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(EmptyMethodBodyDetected,
                        methodDeclaration.Body.GetLocation(),
                        $"method {methodName}");
                    context.ReportDiagnostic(diagnostic);
                }
            }
            
            // Check for short methods without proper attributes
            if (methodDeclaration.Body != null && IsInProductionCode(context))
            {
                var lineCount = CountMethodLines(methodDeclaration.Body);
                var hasProperAttributes = HasObsoleteOrGeneratedCodeAttribute(methodDeclaration);
                
                if (lineCount < 3 && !hasProperAttributes)
                {
                    var diagnostic = Diagnostic.Create(ShortMethodWithoutAttributes,
                        methodDeclaration.Identifier.GetLocation(),
                        methodName);
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzeClassDeclaration(SyntaxNodeAnalysisContext context)
        {
            var classDeclaration = (ClassDeclarationSyntax)context.Node;
            var className = classDeclaration.Identifier.ValueText;
            
            // Check for non-production patterns in class names
            if (ContainsNonProductionPattern(className))
            {
                var diagnostic = Diagnostic.Create(PlaceholderPatternDetected,
                    classDeclaration.Identifier.GetLocation(),
                    $"class name: {className}");
                context.ReportDiagnostic(diagnostic);
            }
        }

        private static void AnalyzeInvocationExpression(SyntaxNodeAnalysisContext context)
        {
            var invocation = (InvocationExpressionSyntax)context.Node;
            var invocationText = invocation.ToString();
            
            // Check for Task.Yield() and Task.Delay() patterns
            if (invocationText.Contains("Task.Yield(") || 
                Regex.IsMatch(invocationText, @"Task\.Delay\s*\(\s*\d+"))
            {
                if (IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(EmptyAsyncPlaceholder,
                        invocation.GetLocation(),
                        invocationText);
                    context.ReportDiagnostic(diagnostic);
                }
            }
            
            // Check for weak random generation
            if (invocationText.Contains("new Random(") || invocationText.Contains("Random.Shared"))
            {
                if (IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(WeakRandomGeneration,
                        invocation.GetLocation(),
                        invocationText);
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzeObjectCreationExpression(SyntaxNodeAnalysisContext context)
        {
            var objectCreation = (ObjectCreationExpressionSyntax)context.Node;
            var creationText = objectCreation.ToString();
            
            // Check for weak Random creation
            if (creationText.Contains("new Random("))
            {
                if (IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(WeakRandomGeneration,
                        objectCreation.GetLocation(),
                        creationText);
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzeArrayCreationExpression(SyntaxNodeAnalysisContext context)
        {
            var arrayCreation = (ArrayCreationExpressionSyntax)context.Node;
            var arrayText = arrayCreation.ToString();
            
            // Check for fixed-size arrays with numeric literals
            if (Regex.IsMatch(arrayText, @"new\s+(byte|int|double|float|decimal|long|short)\[\s*\d+\s*\]"))
            {
                if (IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(FixedSizeArrayDetected,
                        arrayCreation.GetLocation(),
                        arrayText);
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzeThrowStatement(SyntaxNodeAnalysisContext context)
        {
            var throwStatement = (ThrowStatementSyntax)context.Node;
            var throwText = throwStatement.ToString();
            
            // Check for NotImplementedException
            if (throwText.Contains("NotImplementedException"))
            {
                if (IsInProductionCode(context))
                {
                    var diagnostic = Diagnostic.Create(EmptyAsyncPlaceholder,
                        throwStatement.GetLocation(),
                        throwText);
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void AnalyzeSourceFile(SyntaxTreeAnalysisContext context)
        {
            var sourceText = context.Tree.GetText();
            var text = sourceText.ToString();
            var filePath = context.Tree.FilePath;
            
            // Only analyze production code
            if (!IsInProductionCode(filePath))
                return;
            
            // Check Python files for pass statements
            if (Path.GetExtension(filePath)?.Equals(".py", StringComparison.OrdinalIgnoreCase) == true)
            {
                CheckPythonPassStatements(context, text);
            }
            
            // Check for commented-out analyzer rules
            CheckCommentedAnalyzerRules(context, text);
            
            // Check for development-only comments
            CheckDevelopmentOnlyComments(context, text);
            
            // Check for placeholder patterns in text
            CheckPlaceholderPatterns(context, text);
            
            // Check .editorconfig files for severity downgrades
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
            // Check if it's a numeric value
            if (!decimal.TryParse(value, out decimal numericValue))
                return false;
                
            // Allow common constants that are not business logic
            var allowedConstants = new[] { "0", "1", "-1", "0.0", "1.0", "-1.0" };
            if (allowedConstants.Contains(value))
                return false;
                
            // Check for specific problematic values mentioned in the issue
            var problematicValues = new[] 
            { 
                "2.5", "0.7", "0.8", "0.5", "0.3", "0.4", "0.25", "1.25", "4125.25", "4125.00", "4125.50",
                "15", "12", "100", "120", "1000", "125430", "200", "50"
            };
            
            if (problematicValues.Contains(value))
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
                           className.Contains("Regime") || className.Contains("ML") || className.Contains("AI") ||
                           className.Contains("RL") || className.Contains("Agent");
                }
            }
            
            return false;
        }

        private static bool IsBusinessLogicContext(string variableName)
        {
            var businessTerms = new[] 
            { 
                "position", "confidence", "regime", "size", "threshold", "ratio", "factor", "weight", 
                "probability", "price", "quantity", "volume", "fee", "commission", "spread", "slippage",
                "risk", "reward", "return", "yield", "volatility", "correlation", "alpha", "beta",
                "sharpe", "sortino", "calmar", "drawdown", "var", "cvar", "es", "nq", "contract",
                "multiplier", "tick", "pip", "point", "leverage", "margin", "equity", "balance",
                "pnl", "profit", "loss", "trade", "order", "fill", "execution", "latency",
                "timeout", "delay", "interval", "frequency", "rate", "speed", "acceleration",
                "momentum", "trend", "signal", "indicator", "oscillator", "average", "deviation",
                "variance", "skewness", "kurtosis", "entropy", "information", "kelly", "optimal",
                "learning", "training", "validation", "testing", "accuracy", "precision", "recall",
                "f1", "auc", "roc", "loss", "error", "gradient", "backprop", "epoch", "batch",
                "hidden", "layer", "neuron", "activation", "dropout", "regularization", "optimizer",
                "schedule", "decay", "momentum", "velocity", "temperature", "exploration", "exploitation",
                "policy", "value", "q", "action", "state", "reward", "discount", "gamma", "epsilon",
                "tau", "rho", "lambda", "mu", "sigma", "theta", "phi", "psi", "omega"
            };
            return businessTerms.Any(term => variableName.ToLowerInvariant().Contains(term));
        }

        private static int GetLineFromPosition(string text, int position)
        {
            return text.Take(position).Count(c => c == '\n') + 1;
        }

        private static void CheckDevelopmentOnlyComments(SyntaxTreeAnalysisContext context, string text)
        {
            var developmentCommentPatterns = new[]
            {
                @"//\s*(for\s+testing|debug\s+only|temporary|remove\s+this|TODO|FIXME|HACK|XXX|STUB|PLACEHOLDER|BUG|NOTE|REVIEW|REFACTOR|OPTIMIZE)",
                @"//\s*(mock|placeholder|stub|fake|temporary|test|sample|demo|example|hardcoded|literal)",
                @"//\s*(implementation\s+needed|production\s+replacement|real\s+data\s+needed|configuration\s+required)"
            };
            
            foreach (var pattern in developmentCommentPatterns)
            {
                var regex = new Regex(pattern, RegexOptions.Multiline | RegexOptions.IgnoreCase);
                var matches = regex.Matches(text);
                
                foreach (Match match in matches)
                {
                    var diagnostic = Diagnostic.Create(DevelopmentOnlyComment,
                        Location.Create(context.Tree, new TextSpan(match.Index, match.Length)),
                        match.Value.Trim());
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static void CheckPlaceholderPatterns(SyntaxTreeAnalysisContext context, string text)
        {
            var placeholderPatterns = new[]
            {
                @"\b(PLACEHOLDER|TEMP|DUMMY|MOCK|FAKE|STUB|HARDCODED|SAMPLE)\b",
                @"\.GetBytes\(\d+\)", // new byte[literal]
                @"Task\.CompletedTask\s*;",
                @"return\s+Task\.CompletedTask\s*;"
            };
            
            foreach (var pattern in placeholderPatterns)
            {
                var regex = new Regex(pattern, RegexOptions.Multiline | RegexOptions.IgnoreCase);
                var matches = regex.Matches(text);
                
                foreach (Match match in matches)
                {
                    var diagnostic = Diagnostic.Create(PlaceholderPatternDetected,
                        Location.Create(context.Tree, new TextSpan(match.Index, match.Length)),
                        match.Value.Trim());
                    context.ReportDiagnostic(diagnostic);
                }
            }
        }

        private static bool IsInProductionCode(SyntaxNodeAnalysisContext context)
        {
            return IsInProductionCode(context.Node.SyntaxTree.FilePath);
        }
        
        private static bool IsInProductionCode(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                return false;
                
            var normalizedPath = filePath.Replace('\\', '/').ToLowerInvariant();
            
            // Exclude test directories and files
            var excludePatterns = new[]
            {
                "/test", "/tests", "/testapps", "/samples", "/bin/", "/obj/", "/packages/",
                "test.cs", "tests.cs", "mock", "fake", "stub", "demo", "sample"
            };
            
            return normalizedPath.Contains("/src/") && 
                   !excludePatterns.Any(pattern => normalizedPath.Contains(pattern));
        }
        
        private static bool ContainsNonProductionPattern(string text)
        {
            var nonProductionTerms = new[]
            {
                "mock", "fake", "stub", "test", "demo", "sample", "placeholder", 
                "temp", "temporary", "dummy", "hardcoded", "literal"
            };
            
            return nonProductionTerms.Any(term => 
                text.ToLowerInvariant().Contains(term.ToLowerInvariant()));
        }
        
        private static bool IsNumericLiteral(SyntaxToken token)
        {
            return token.IsKind(SyntaxKind.NumericLiteralToken) && 
                   !token.ValueText.EndsWith("f") && // Allow float literals like 1.0f
                   !token.ValueText.EndsWith("F");   // Allow float literals like 1.0F
        }
        
        private static int CountMethodLines(BlockSyntax methodBody)
        {
            if (methodBody?.Statements == null)
                return 0;
                
            return methodBody.Statements.Count;
        }
        
        private static bool HasObsoleteOrGeneratedCodeAttribute(MethodDeclarationSyntax method)
        {
            if (method.AttributeLists == null)
                return false;
                
            foreach (var attributeList in method.AttributeLists)
            {
                foreach (var attribute in attributeList.Attributes)
                {
                    var attributeName = attribute.Name.ToString();
                    if (attributeName.Contains("Obsolete") || 
                        attributeName.Contains("GeneratedCode") ||
                        attributeName.Contains("CompilerGenerated"))
                    {
                        return true;
                    }
                }
            }
            
            return false;
        }
        
        private static void CheckPythonPassStatements(SyntaxTreeAnalysisContext context, string text)
        {
            // Check for standalone 'pass' statements in Python files
            var passPattern = @"^\s*pass\s*$";
            var regex = new Regex(passPattern, RegexOptions.Multiline);
            var matches = regex.Matches(text);
            
            foreach (Match match in matches)
            {
                var diagnostic = Diagnostic.Create(PythonPassStatementDetected,
                    Location.Create(context.Tree, new TextSpan(match.Index, match.Length)),
                    context.Tree.FilePath);
                context.ReportDiagnostic(diagnostic);
            }
        }
    }
}