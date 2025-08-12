"""
Configuration Management for Master Agent Orchestrator.

This module provides comprehensive configuration management with
environment variable support, validation, and different deployment profiles.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

try:
    from decouple import config, Csv
    DECOUPLE_AVAILABLE = True
except ImportError:
    DECOUPLE_AVAILABLE = False
    # Fallback to os.getenv
    def config(key, default=None, cast=None):
        value = os.getenv(key, default)
        if cast and value is not None:
            return cast(value)
        return value
    
    def Csv(cast=None):
        def csv_cast(value):
            if isinstance(value, str):
                items = [item.strip() for item in value.split(',')]
                if cast:
                    return [cast(item) for item in items]
                return items
            return value
        return csv_cast


class Environment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str
    organization: Optional[str] = None
    project: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'OpenAIConfig':
        """Create OpenAI config from environment variables."""
        return cls(
            api_key=config('OPENAI_API_KEY', default=''),
            organization=config('OPENAI_ORGANIZATION', default=None),
            project=config('OPENAI_PROJECT', default=None),
            base_url=config('OPENAI_BASE_URL', default=None),
            timeout=config('OPENAI_TIMEOUT', default=30.0, cast=float),
            max_retries=config('OPENAI_MAX_RETRIES', default=3, cast=int)
        )


@dataclass
class OrchestratorConfig:
    """Core orchestrator configuration."""
    max_concurrent_agents: int = 10
    default_model: str = "gpt-4o"
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    task_timeout: float = 300.0  # 5 minutes
    agent_idle_timeout: float = 3600.0  # 1 hour
    
    @classmethod
    def from_env(cls) -> 'OrchestratorConfig':
        """Create orchestrator config from environment variables."""
        return cls(
            max_concurrent_agents=config('MAX_CONCURRENT_AGENTS', default=10, cast=int),
            default_model=config('DEFAULT_MODEL', default='gpt-4o'),
            enable_logging=config('ENABLE_LOGGING', default=True, cast=bool),
            log_level=config('LOG_LEVEL', default='INFO'),
            log_file=config('LOG_FILE', default=None),
            task_timeout=config('TASK_TIMEOUT', default=300.0, cast=float),
            agent_idle_timeout=config('AGENT_IDLE_TIMEOUT', default=3600.0, cast=float)
        )


@dataclass
class ToolRegistryConfig:
    """Tool registry configuration."""
    enable_metrics: bool = True
    max_tool_history: int = 1000
    default_timeout: float = 60.0
    rate_limit_window: int = 60  # seconds
    builtin_tools_enabled: bool = True
    custom_tools_dir: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'ToolRegistryConfig':
        """Create tool registry config from environment variables."""
        return cls(
            enable_metrics=config('TOOL_ENABLE_METRICS', default=True, cast=bool),
            max_tool_history=config('TOOL_MAX_HISTORY', default=1000, cast=int),
            default_timeout=config('TOOL_DEFAULT_TIMEOUT', default=60.0, cast=float),
            rate_limit_window=config('TOOL_RATE_LIMIT_WINDOW', default=60, cast=int),
            builtin_tools_enabled=config('BUILTIN_TOOLS_ENABLED', default=True, cast=bool),
            custom_tools_dir=config('CUSTOM_TOOLS_DIR', default=None)
        )


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration."""
    enable_circuit_breakers: bool = True
    enable_auto_recovery: bool = True
    max_error_history: int = 1000
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 60.0
    retry_max_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_backoff_factor: float = 2.0
    
    @classmethod
    def from_env(cls) -> 'ErrorHandlingConfig':
        """Create error handling config from environment variables."""
        return cls(
            enable_circuit_breakers=config('ENABLE_CIRCUIT_BREAKERS', default=True, cast=bool),
            enable_auto_recovery=config('ENABLE_AUTO_RECOVERY', default=True, cast=bool),
            max_error_history=config('MAX_ERROR_HISTORY', default=1000, cast=int),
            circuit_failure_threshold=config('CIRCUIT_FAILURE_THRESHOLD', default=5, cast=int),
            circuit_recovery_timeout=config('CIRCUIT_RECOVERY_TIMEOUT', default=60.0, cast=float),
            retry_max_attempts=config('RETRY_MAX_ATTEMPTS', default=3, cast=int),
            retry_initial_delay=config('RETRY_INITIAL_DELAY', default=1.0, cast=float),
            retry_max_delay=config('RETRY_MAX_DELAY', default=60.0, cast=float),
            retry_backoff_factor=config('RETRY_BACKOFF_FACTOR', default=2.0, cast=float)
        )


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_auth: bool = False
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 100
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_cors: bool = True
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Create security config from environment variables."""
        return cls(
            enable_auth=config('ENABLE_AUTH', default=False, cast=bool),
            api_key_header=config('API_KEY_HEADER', default='X-API-Key'),
            allowed_origins=config('ALLOWED_ORIGINS', default=[], cast=Csv()),
            rate_limit_per_minute=config('RATE_LIMIT_PER_MINUTE', default=100, cast=int),
            max_request_size=config('MAX_REQUEST_SIZE', default=10*1024*1024, cast=int),
            enable_cors=config('ENABLE_CORS', default=True, cast=bool)
        )


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    metrics_port: int = 8090
    enable_health_checks: bool = True
    health_check_interval: float = 30.0
    enable_tracing: bool = False
    jaeger_endpoint: Optional[str] = None
    prometheus_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        """Create monitoring config from environment variables."""
        return cls(
            enable_metrics=config('ENABLE_METRICS', default=True, cast=bool),
            metrics_port=config('METRICS_PORT', default=8090, cast=int),
            enable_health_checks=config('ENABLE_HEALTH_CHECKS', default=True, cast=bool),
            health_check_interval=config('HEALTH_CHECK_INTERVAL', default=30.0, cast=float),
            enable_tracing=config('ENABLE_TRACING', default=False, cast=bool),
            jaeger_endpoint=config('JAEGER_ENDPOINT', default=None),
            prometheus_enabled=config('PROMETHEUS_ENABLED', default=True, cast=bool)
        )


@dataclass
class DatabaseConfig:
    """Database configuration (optional)."""
    enabled: bool = False
    url: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: float = 30.0
    echo: bool = False
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create database config from environment variables."""
        return cls(
            enabled=config('DATABASE_ENABLED', default=False, cast=bool),
            url=config('DATABASE_URL', default=None),
            pool_size=config('DB_POOL_SIZE', default=10, cast=int),
            max_overflow=config('DB_MAX_OVERFLOW', default=20, cast=int),
            pool_timeout=config('DB_POOL_TIMEOUT', default=30.0, cast=float),
            echo=config('DB_ECHO', default=False, cast=bool)
        )


@dataclass
class RedisConfig:
    """Redis configuration (optional)."""
    enabled: bool = False
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        """Create Redis config from environment variables."""
        return cls(
            enabled=config('REDIS_ENABLED', default=False, cast=bool),
            url=config('REDIS_URL', default=None),
            host=config('REDIS_HOST', default='localhost'),
            port=config('REDIS_PORT', default=6379, cast=int),
            db=config('REDIS_DB', default=0, cast=int),
            password=config('REDIS_PASSWORD', default=None),
            max_connections=config('REDIS_MAX_CONNECTIONS', default=10, cast=int)
        )


@dataclass
class MasterOrchestratorConfig:
    """Complete configuration for the Master Agent Orchestrator."""
    environment: Environment
    openai: OpenAIConfig
    orchestrator: OrchestratorConfig
    tools: ToolRegistryConfig
    error_handling: ErrorHandlingConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    database: DatabaseConfig
    redis: RedisConfig
    
    @classmethod
    def from_env(cls, env: Optional[str] = None) -> 'MasterOrchestratorConfig':
        """Create complete config from environment variables."""
        environment = Environment(env or config('ENVIRONMENT', default='development'))
        
        return cls(
            environment=environment,
            openai=OpenAIConfig.from_env(),
            orchestrator=OrchestratorConfig.from_env(),
            tools=ToolRegistryConfig.from_env(),
            error_handling=ErrorHandlingConfig.from_env(),
            security=SecurityConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env()
        )
    
    @classmethod
    def from_file(cls, config_file: str) -> 'MasterOrchestratorConfig':
        """Create config from JSON/YAML file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML configuration files")
            else:
                data = json.load(f)
        
        # Convert dict to config objects
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'MasterOrchestratorConfig':
        """Create config from dictionary."""
        env = Environment(data.get('environment', 'development'))
        
        return cls(
            environment=env,
            openai=OpenAIConfig(**data.get('openai', {})),
            orchestrator=OrchestratorConfig(**data.get('orchestrator', {})),
            tools=ToolRegistryConfig(**data.get('tools', {})),
            error_handling=ErrorHandlingConfig(**data.get('error_handling', {})),
            security=SecurityConfig(**data.get('security', {})),
            monitoring=MonitoringConfig(**data.get('monitoring', {})),
            database=DatabaseConfig(**data.get('database', {})),
            redis=RedisConfig(**data.get('redis', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'environment': self.environment.value,
            'openai': {
                'api_key': self.openai.api_key,
                'organization': self.openai.organization,
                'project': self.openai.project,
                'base_url': self.openai.base_url,
                'timeout': self.openai.timeout,
                'max_retries': self.openai.max_retries
            },
            'orchestrator': {
                'max_concurrent_agents': self.orchestrator.max_concurrent_agents,
                'default_model': self.orchestrator.default_model,
                'enable_logging': self.orchestrator.enable_logging,
                'log_level': self.orchestrator.log_level,
                'log_file': self.orchestrator.log_file,
                'task_timeout': self.orchestrator.task_timeout,
                'agent_idle_timeout': self.orchestrator.agent_idle_timeout
            },
            'tools': {
                'enable_metrics': self.tools.enable_metrics,
                'max_tool_history': self.tools.max_tool_history,
                'default_timeout': self.tools.default_timeout,
                'rate_limit_window': self.tools.rate_limit_window,
                'builtin_tools_enabled': self.tools.builtin_tools_enabled,
                'custom_tools_dir': self.tools.custom_tools_dir
            },
            'error_handling': {
                'enable_circuit_breakers': self.error_handling.enable_circuit_breakers,
                'enable_auto_recovery': self.error_handling.enable_auto_recovery,
                'max_error_history': self.error_handling.max_error_history,
                'circuit_failure_threshold': self.error_handling.circuit_failure_threshold,
                'circuit_recovery_timeout': self.error_handling.circuit_recovery_timeout,
                'retry_max_attempts': self.error_handling.retry_max_attempts,
                'retry_initial_delay': self.error_handling.retry_initial_delay,
                'retry_max_delay': self.error_handling.retry_max_delay,
                'retry_backoff_factor': self.error_handling.retry_backoff_factor
            },
            'security': {
                'enable_auth': self.security.enable_auth,
                'api_key_header': self.security.api_key_header,
                'allowed_origins': self.security.allowed_origins,
                'rate_limit_per_minute': self.security.rate_limit_per_minute,
                'max_request_size': self.security.max_request_size,
                'enable_cors': self.security.enable_cors
            },
            'monitoring': {
                'enable_metrics': self.monitoring.enable_metrics,
                'metrics_port': self.monitoring.metrics_port,
                'enable_health_checks': self.monitoring.enable_health_checks,
                'health_check_interval': self.monitoring.health_check_interval,
                'enable_tracing': self.monitoring.enable_tracing,
                'jaeger_endpoint': self.monitoring.jaeger_endpoint,
                'prometheus_enabled': self.monitoring.prometheus_enabled
            },
            'database': {
                'enabled': self.database.enabled,
                'url': self.database.url,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout,
                'echo': self.database.echo
            },
            'redis': {
                'enabled': self.redis.enabled,
                'url': self.redis.url,
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'password': self.redis.password,
                'max_connections': self.redis.max_connections
            }
        }
    
    def save_to_file(self, config_file: str) -> None:
        """Save config to JSON/YAML file."""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    yaml.safe_dump(data, f, default_flow_style=False, indent=2)
                except ImportError:
                    raise ImportError("PyYAML required for YAML configuration files")
            else:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # OpenAI validation
        if not self.openai.api_key:
            issues.append("OpenAI API key is required")
        
        if self.openai.timeout <= 0:
            issues.append("OpenAI timeout must be positive")
        
        # Orchestrator validation
        if self.orchestrator.max_concurrent_agents <= 0:
            issues.append("Max concurrent agents must be positive")
        
        if self.orchestrator.task_timeout <= 0:
            issues.append("Task timeout must be positive")
        
        # Tool registry validation
        if self.tools.max_tool_history <= 0:
            issues.append("Max tool history must be positive")
        
        if self.tools.default_timeout <= 0:
            issues.append("Default tool timeout must be positive")
        
        # Error handling validation
        if self.error_handling.max_error_history <= 0:
            issues.append("Max error history must be positive")
        
        if self.error_handling.retry_max_attempts <= 0:
            issues.append("Retry max attempts must be positive")
        
        # Security validation
        if self.security.rate_limit_per_minute <= 0:
            issues.append("Rate limit per minute must be positive")
        
        if self.security.max_request_size <= 0:
            issues.append("Max request size must be positive")
        
        # Monitoring validation
        if self.monitoring.metrics_port <= 0 or self.monitoring.metrics_port > 65535:
            issues.append("Metrics port must be between 1 and 65535")
        
        # Database validation
        if self.database.enabled and not self.database.url:
            issues.append("Database URL is required when database is enabled")
        
        # Redis validation
        if self.redis.enabled and not self.redis.url and not self.redis.host:
            issues.append("Redis URL or host is required when Redis is enabled")
        
        if self.redis.port <= 0 or self.redis.port > 65535:
            issues.append("Redis port must be between 1 and 65535")
        
        return issues
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def get_profile_configs(self) -> Dict[str, 'MasterOrchestratorConfig']:
        """Get predefined configuration profiles."""
        base_config = self.to_dict()
        
        # Development profile
        dev_config = base_config.copy()
        dev_config.update({
            'environment': 'development',
            'orchestrator': {
                **dev_config['orchestrator'],
                'max_concurrent_agents': 3,
                'enable_logging': True,
                'log_level': 'DEBUG'
            },
            'error_handling': {
                **dev_config['error_handling'],
                'enable_auto_recovery': True,
                'max_error_history': 100
            },
            'monitoring': {
                **dev_config['monitoring'],
                'enable_metrics': True,
                'enable_health_checks': False
            }
        })
        
        # Production profile
        prod_config = base_config.copy()
        prod_config.update({
            'environment': 'production',
            'orchestrator': {
                **prod_config['orchestrator'],
                'max_concurrent_agents': 20,
                'enable_logging': True,
                'log_level': 'INFO'
            },
            'error_handling': {
                **prod_config['error_handling'],
                'enable_auto_recovery': True,
                'max_error_history': 5000
            },
            'monitoring': {
                **prod_config['monitoring'],
                'enable_metrics': True,
                'enable_health_checks': True,
                'prometheus_enabled': True
            },
            'security': {
                **prod_config['security'],
                'enable_auth': True,
                'rate_limit_per_minute': 1000
            }
        })
        
        return {
            'development': self._from_dict(dev_config),
            'production': self._from_dict(prod_config)
        }