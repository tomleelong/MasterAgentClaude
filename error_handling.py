"""
Comprehensive Error Handling and Recovery System for Agent Orchestrator.

This module provides robust error handling, recovery mechanisms, and
resilience patterns for multi-agent systems.
"""

import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import functools


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    AGENT_ERROR = "agent_error"
    TOOL_ERROR = "tool_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ESCALATE = "escalate"
    RESET = "reset"
    TERMINATE = "terminate"


@dataclass
class ErrorContext:
    """Context information for error analysis."""
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    tool_name: Optional[str] = None
    operation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Comprehensive error record."""
    id: str
    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    stack_trace: str
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    resolved_at: Optional[datetime] = None
    occurrences: int = 1
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count_after_half_open: int = 0
    required_successes: int = 3


class AgentOrchestratorError(Exception):
    """Base exception for orchestrator-specific errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.recoverable = recoverable


class AgentCreationError(AgentOrchestratorError):
    """Error during agent creation."""
    
    def __init__(self, message: str, agent_config: Optional[Dict] = None):
        super().__init__(
            message,
            category=ErrorCategory.AGENT_ERROR,
            severity=ErrorSeverity.HIGH
        )
        if agent_config:
            self.context.metadata["agent_config"] = agent_config


class ToolExecutionError(AgentOrchestratorError):
    """Error during tool execution."""
    
    def __init__(self, message: str, tool_name: str, agent_id: Optional[str] = None):
        context = ErrorContext(agent_id=agent_id, tool_name=tool_name)
        super().__init__(
            message,
            category=ErrorCategory.TOOL_ERROR,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )


class TaskExecutionError(AgentOrchestratorError):
    """Error during task execution."""
    
    def __init__(self, message: str, task_id: str, agent_id: Optional[str] = None):
        context = ErrorContext(agent_id=agent_id, task_id=task_id)
        super().__init__(
            message,
            category=ErrorCategory.AGENT_ERROR,
            severity=ErrorSeverity.HIGH,
            context=context
        )


class ResourceExhaustionError(AgentOrchestratorError):
    """Error due to resource exhaustion."""
    
    def __init__(self, message: str, resource_type: str):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE_ERROR,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False
        )
        self.context.metadata["resource_type"] = resource_type


class ErrorHandler:
    """
    Comprehensive error handling and recovery system.
    
    Provides error classification, recovery strategies, circuit breakers,
    and resilience patterns for the agent orchestrator.
    """
    
    def __init__(
        self,
        enable_circuit_breakers: bool = True,
        enable_auto_recovery: bool = True,
        max_error_history: int = 1000
    ):
        """
        Initialize the error handler.
        
        Args:
            enable_circuit_breakers: Whether to use circuit breakers
            enable_auto_recovery: Whether to attempt automatic recovery
            max_error_history: Maximum number of errors to keep in history
        """
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_auto_recovery = enable_auto_recovery
        self.max_error_history = max_error_history
        
        # Error tracking
        self.error_history: Dict[str, ErrorRecord] = {}
        self.error_patterns: Dict[str, int] = {}  # Pattern -> count
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[ErrorCategory, RecoveryStrategy] = {
            ErrorCategory.NETWORK_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.TIMEOUT_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.TOOL_ERROR: RecoveryStrategy.FALLBACK,
            ErrorCategory.PERMISSION_ERROR: RecoveryStrategy.ESCALATE,
            ErrorCategory.VALIDATION_ERROR: RecoveryStrategy.SKIP,
            ErrorCategory.RESOURCE_ERROR: RecoveryStrategy.RESET,
            ErrorCategory.SYSTEM_ERROR: RecoveryStrategy.TERMINATE
        }
        
        # Retry configurations
        self.retry_configs: Dict[ErrorCategory, RetryConfig] = {
            ErrorCategory.NETWORK_ERROR: RetryConfig(max_attempts=5, initial_delay=1.0),
            ErrorCategory.TIMEOUT_ERROR: RetryConfig(max_attempts=3, initial_delay=2.0),
            ErrorCategory.TOOL_ERROR: RetryConfig(max_attempts=2, initial_delay=0.5),
            ErrorCategory.AGENT_ERROR: RetryConfig(max_attempts=3, initial_delay=1.0)
        }
        
        self.logger = logging.getLogger(__name__)
    
    def classify_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None
    ) -> ErrorCategory:
        """
        Classify an error into appropriate category.
        
        Args:
            error: Exception to classify
            context: Error context
            
        Returns:
            Error category
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Check for custom orchestrator errors
        if isinstance(error, AgentOrchestratorError):
            return error.category
        
        # Network-related errors
        if any(keyword in error_message for keyword in [
            'connection', 'network', 'timeout', 'dns', 'socket'
        ]) or error_type in ['ConnectionError', 'TimeoutError', 'NetworkError']:
            return ErrorCategory.NETWORK_ERROR
        
        # Permission errors
        if any(keyword in error_message for keyword in [
            'permission', 'access denied', 'unauthorized', 'forbidden'
        ]) or error_type in ['PermissionError', 'AccessDenied']:
            return ErrorCategory.PERMISSION_ERROR
        
        # Validation errors
        if any(keyword in error_message for keyword in [
            'validation', 'invalid', 'malformed', 'schema'
        ]) or error_type in ['ValueError', 'ValidationError']:
            return ErrorCategory.VALIDATION_ERROR
        
        # Resource errors
        if any(keyword in error_message for keyword in [
            'memory', 'disk space', 'quota', 'limit exceeded'
        ]) or error_type in ['MemoryError', 'ResourceExhausted']:
            return ErrorCategory.RESOURCE_ERROR
        
        # Tool-related errors (check context)
        if context and context.tool_name:
            return ErrorCategory.TOOL_ERROR
        
        # Agent-related errors (check context)
        if context and context.agent_id:
            return ErrorCategory.AGENT_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def determine_severity(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Optional[ErrorContext] = None
    ) -> ErrorSeverity:
        """
        Determine error severity based on type and context.
        
        Args:
            error: Exception
            category: Error category
            context: Error context
            
        Returns:
            Error severity
        """
        # Check for custom severity
        if isinstance(error, AgentOrchestratorError):
            return error.severity
        
        # Critical severity mapping
        critical_categories = [ErrorCategory.RESOURCE_ERROR, ErrorCategory.SYSTEM_ERROR]
        if category in critical_categories:
            return ErrorSeverity.CRITICAL
        
        # High severity for certain error types
        high_severity_types = ['SystemExit', 'KeyboardInterrupt', 'MemoryError']
        if type(error).__name__ in high_severity_types:
            return ErrorSeverity.CRITICAL
        
        # Context-based severity
        if context:
            # Critical if affects master orchestrator
            if context.operation in ['create_agent', 'shutdown', 'initialization']:
                return ErrorSeverity.HIGH
        
        # Default mappings
        severity_map = {
            ErrorCategory.NETWORK_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.TIMEOUT_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.TOOL_ERROR: ErrorSeverity.LOW,
            ErrorCategory.PERMISSION_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.VALIDATION_ERROR: ErrorSeverity.LOW,
            ErrorCategory.AGENT_ERROR: ErrorSeverity.HIGH
        }
        
        return severity_map.get(category, ErrorSeverity.MEDIUM)
    
    async def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        auto_recover: bool = None
    ) -> Optional[Any]:
        """
        Handle an error with classification, logging, and recovery.
        
        Args:
            error: Exception to handle
            context: Error context
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        if auto_recover is None:
            auto_recover = self.enable_auto_recovery
        
        try:
            # Classify error
            category = self.classify_error(error, context)
            severity = self.determine_severity(error, category, context)
            
            # Create error record
            error_id = self._generate_error_id(error, context)
            if error_id in self.error_history:
                # Update existing record
                record = self.error_history[error_id]
                record.occurrences += 1
                record.last_occurrence = datetime.now()
            else:
                # Create new record
                record = ErrorRecord(
                    id=error_id,
                    error=error,
                    category=category,
                    severity=severity,
                    context=context or ErrorContext(),
                    stack_trace=traceback.format_exc()
                )
                self.error_history[error_id] = record
            
            # Log error
            self._log_error(record)
            
            # Update error patterns
            pattern = self._extract_error_pattern(error)
            self.error_patterns[pattern] = self.error_patterns.get(pattern, 0) + 1
            
            # Check circuit breaker
            if self.enable_circuit_breakers:
                circuit_name = self._get_circuit_name(context)
                if circuit_name and not self._check_circuit_breaker(circuit_name):
                    raise AgentOrchestratorError(
                        f"Circuit breaker {circuit_name} is open",
                        category=ErrorCategory.SYSTEM_ERROR,
                        severity=ErrorSeverity.HIGH,
                        recoverable=False
                    )
            
            # Attempt recovery if enabled
            if auto_recover and record.severity != ErrorSeverity.CRITICAL:
                recovery_result = await self._attempt_recovery(record)
                if recovery_result is not None:
                    record.recovery_successful = True
                    record.resolved_at = datetime.now()
                    return recovery_result
            
            # Clean up error history if needed
            await self._cleanup_error_history()
            
            return None
            
        except Exception as handler_error:
            self.logger.critical(f"Error handler failed: {str(handler_error)}")
            raise error  # Re-raise original error
    
    async def _attempt_recovery(self, record: ErrorRecord) -> Optional[Any]:
        """
        Attempt to recover from an error using appropriate strategy.
        
        Args:
            record: Error record
            
        Returns:
            Recovery result if successful
        """
        if record.recovery_attempted:
            return None
        
        record.recovery_attempted = True
        strategy = self.recovery_strategies.get(record.category, RecoveryStrategy.ESCALATE)
        record.recovery_strategy = strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._retry_operation(record)
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._fallback_operation(record)
            elif strategy == RecoveryStrategy.RESET:
                return await self._reset_operation(record)
            elif strategy == RecoveryStrategy.SKIP:
                self.logger.info(f"Skipping failed operation: {record.id}")
                return "operation_skipped"
            elif strategy == RecoveryStrategy.ESCALATE:
                await self._escalate_error(record)
            elif strategy == RecoveryStrategy.TERMINATE:
                await self._terminate_operation(record)
            
            return None
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed for {record.id}: {str(recovery_error)}")
            return None
    
    async def _retry_operation(self, record: ErrorRecord) -> Optional[Any]:
        """Retry failed operation with exponential backoff."""
        category = record.category
        retry_config = self.retry_configs.get(category, RetryConfig())
        
        for attempt in range(1, retry_config.max_attempts + 1):
            if attempt > 1:  # Don't delay on first attempt
                delay = min(
                    retry_config.initial_delay * (retry_config.backoff_factor ** (attempt - 2)),
                    retry_config.max_delay
                )
                
                if retry_config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)  # ±50% jitter
                
                await asyncio.sleep(delay)
            
            try:
                # This would need to be implemented based on the specific operation
                # For now, we'll simulate a retry
                self.logger.info(f"Retry attempt {attempt}/{retry_config.max_attempts} for {record.id}")
                
                # TODO: Implement actual retry logic based on context
                # This would involve re-executing the failed operation
                
                return f"retry_successful_attempt_{attempt}"
                
            except Exception as retry_error:
                if attempt == retry_config.max_attempts:
                    self.logger.error(f"All retry attempts failed for {record.id}")
                    raise retry_error
                else:
                    self.logger.warning(f"Retry attempt {attempt} failed for {record.id}: {str(retry_error)}")
        
        return None
    
    async def _fallback_operation(self, record: ErrorRecord) -> Optional[Any]:
        """Execute fallback operation."""
        self.logger.info(f"Executing fallback for {record.id}")
        
        # TODO: Implement fallback logic based on context
        # This could involve using alternative tools, agents, or methods
        
        if record.context.tool_name:
            # Tool fallback - could use alternative tool
            return f"fallback_tool_used_for_{record.context.tool_name}"
        elif record.context.agent_id:
            # Agent fallback - could reassign task to different agent
            return f"fallback_agent_assigned_for_{record.context.agent_id}"
        
        return "fallback_executed"
    
    async def _reset_operation(self, record: ErrorRecord) -> Optional[Any]:
        """Reset system state to recover from error."""
        self.logger.info(f"Resetting system state for {record.id}")
        
        # TODO: Implement reset logic based on context
        # This could involve clearing caches, restarting services, etc.
        
        return "system_reset_completed"
    
    async def _escalate_error(self, record: ErrorRecord) -> None:
        """Escalate error to higher-level handlers."""
        self.logger.warning(f"Escalating error {record.id} (severity: {record.severity.value})")
        
        # TODO: Implement escalation logic
        # This could involve notifying administrators, triggering alerts, etc.
    
    async def _terminate_operation(self, record: ErrorRecord) -> None:
        """Terminate operation due to critical error."""
        self.logger.critical(f"Terminating operation due to critical error {record.id}")
        
        # TODO: Implement termination logic
        # This could involve shutting down agents, cleaning up resources, etc.
    
    def _check_circuit_breaker(self, circuit_name: str) -> bool:
        """
        Check if circuit breaker allows operation.
        
        Args:
            circuit_name: Circuit breaker name
            
        Returns:
            True if operation allowed
        """
        if circuit_name not in self.circuit_breakers:
            self.circuit_breakers[circuit_name] = CircuitBreaker(name=circuit_name)
        
        breaker = self.circuit_breakers[circuit_name]
        now = datetime.now()
        
        if breaker.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (breaker.last_failure_time and 
                (now - breaker.last_failure_time).total_seconds() >= breaker.recovery_timeout):
                breaker.state = CircuitBreakerState.HALF_OPEN
                breaker.success_count_after_half_open = 0
                self.logger.info(f"Circuit breaker {circuit_name} moved to HALF_OPEN")
            else:
                return False
        
        return True
    
    def record_circuit_success(self, circuit_name: str) -> None:
        """Record successful operation for circuit breaker."""
        if circuit_name in self.circuit_breakers:
            breaker = self.circuit_breakers[circuit_name]
            
            if breaker.state == CircuitBreakerState.HALF_OPEN:
                breaker.success_count_after_half_open += 1
                if breaker.success_count_after_half_open >= breaker.required_successes:
                    breaker.state = CircuitBreakerState.CLOSED
                    breaker.failure_count = 0
                    self.logger.info(f"Circuit breaker {circuit_name} moved to CLOSED")
            elif breaker.state == CircuitBreakerState.CLOSED:
                breaker.failure_count = max(0, breaker.failure_count - 1)
    
    def record_circuit_failure(self, circuit_name: str) -> None:
        """Record failed operation for circuit breaker."""
        if circuit_name not in self.circuit_breakers:
            self.circuit_breakers[circuit_name] = CircuitBreaker(name=circuit_name)
        
        breaker = self.circuit_breakers[circuit_name]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {circuit_name} moved to OPEN")
    
    def _get_circuit_name(self, context: Optional[ErrorContext]) -> Optional[str]:
        """Get circuit breaker name based on context."""
        if not context:
            return None
        
        if context.tool_name:
            return f"tool_{context.tool_name}"
        elif context.agent_id:
            return f"agent_{context.agent_id}"
        elif context.operation:
            return f"operation_{context.operation}"
        
        return None
    
    def _generate_error_id(self, error: Exception, context: Optional[ErrorContext]) -> str:
        """Generate unique error ID."""
        import hashlib
        
        components = [
            type(error).__name__,
            str(error)[:100],  # Truncate long messages
            context.agent_id or "",
            context.tool_name or "",
            context.operation or ""
        ]
        
        return hashlib.md5("|".join(components).encode()).hexdigest()
    
    def _extract_error_pattern(self, error: Exception) -> str:
        """Extract error pattern for analysis."""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Extract common patterns
        if "timeout" in error_msg.lower():
            return f"{error_type}_timeout"
        elif "connection" in error_msg.lower():
            return f"{error_type}_connection"
        elif "permission" in error_msg.lower():
            return f"{error_type}_permission"
        
        return error_type
    
    def _log_error(self, record: ErrorRecord) -> None:
        """Log error with appropriate level."""
        log_msg = (
            f"Error {record.id}: {record.error} "
            f"(Category: {record.category.value}, Severity: {record.severity.value})"
        )
        
        if record.context.agent_id:
            log_msg += f" [Agent: {record.context.agent_id}]"
        if record.context.tool_name:
            log_msg += f" [Tool: {record.context.tool_name}]"
        if record.context.task_id:
            log_msg += f" [Task: {record.context.task_id}]"
        
        if record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_msg)
        elif record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_msg)
        elif record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
    
    async def _cleanup_error_history(self) -> None:
        """Clean up old error history entries."""
        if len(self.error_history) <= self.max_error_history:
            return
        
        # Remove oldest entries
        sorted_errors = sorted(
            self.error_history.items(),
            key=lambda x: x[1].last_occurrence
        )
        
        to_remove = len(self.error_history) - self.max_error_history
        for error_id, _ in sorted_errors[:to_remove]:
            del self.error_history[error_id]
        
        self.logger.info(f"Cleaned up {to_remove} old error records")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        
        stats = {
            "total_errors": len(self.error_history),
            "error_categories": {},
            "error_severities": {},
            "recent_errors": 0,
            "top_error_patterns": {},
            "circuit_breaker_states": {},
            "recovery_success_rate": 0
        }
        
        # Analyze errors
        recovered_count = 0
        for record in self.error_history.values():
            # Categories
            category = record.category.value
            stats["error_categories"][category] = stats["error_categories"].get(category, 0) + 1
            
            # Severities
            severity = record.severity.value
            stats["error_severities"][severity] = stats["error_severities"].get(severity, 0) + 1
            
            # Recent errors
            if record.last_occurrence >= day_ago:
                stats["recent_errors"] += 1
            
            # Recovery rate
            if record.recovery_successful:
                recovered_count += 1
        
        # Recovery success rate
        if len(self.error_history) > 0:
            stats["recovery_success_rate"] = recovered_count / len(self.error_history)
        
        # Top error patterns
        sorted_patterns = sorted(
            self.error_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        stats["top_error_patterns"] = dict(sorted_patterns[:10])
        
        # Circuit breaker states
        for name, breaker in self.circuit_breakers.items():
            stats["circuit_breaker_states"][name] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count
            }
        
        return stats


# Decorator for automatic error handling
def with_error_handling(
    error_handler: ErrorHandler,
    context_factory: Optional[Callable] = None,
    auto_recover: bool = True
):
    """
    Decorator to automatically handle errors in functions.
    
    Args:
        error_handler: ErrorHandler instance
        context_factory: Function to create error context
        auto_recover: Whether to attempt automatic recovery
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = context_factory(*args, **kwargs) if context_factory else None
                await error_handler.handle_error(e, context, auto_recover)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = context_factory(*args, **kwargs) if context_factory else None
                asyncio.create_task(error_handler.handle_error(e, context, auto_recover))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator