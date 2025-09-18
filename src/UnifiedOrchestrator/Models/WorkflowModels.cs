using System.Text.Json.Serialization;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Models;

// Type aliases to Abstractions models for backward compatibility
using UnifiedWorkflow = TradingBot.Abstractions.UnifiedWorkflow;
using WorkflowSchedule = TradingBot.Abstractions.WorkflowSchedule;
using WorkflowType = TradingBot.Abstractions.WorkflowType;
using WorkflowMetrics = TradingBot.Abstractions.WorkflowMetrics;
using WorkflowExecutionContext = TradingBot.Abstractions.WorkflowExecutionContext;
using WorkflowExecutionStatus = TradingBot.Abstractions.WorkflowExecutionStatus;
using WorkflowExecutionResult = TradingBot.Abstractions.WorkflowExecutionResult;
using CloudTradingRecommendation = TradingBot.Abstractions.CloudTradingRecommendation;
