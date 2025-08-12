# 🧠 Ephemeral Session Memory Design

## Overview

This document outlines the design and implementation plan for adding ephemeral session memory support to brain-proxy. This feature enables temporary, user-specific conversation contexts while maintaining persistent tenant knowledge bases.

## Problem Statement

Currently, brain-proxy provides persistent memory per tenant. However, for customer support and multi-user scenarios, we need:
- **Persistent tenant knowledge**: Brand information, policies, product details
- **Ephemeral session memory**: Individual user conversations that don't persist indefinitely

## Solution Architecture

### Tenant ID Format

Tenants can now include an optional session identifier using a colon separator:
- Format: `{base_tenant}:{session_id}`
- Examples:
  - `tenant1:+15551234567` (phone support)
  - `tenant1:user@email.com` (email support)
  - `tenant1:chat_session_xyz` (web chat)

### Memory Separation

```
┌─────────────────────────────────────────┐
│           Full Tenant ID                 │
│        "acme:+15551234567"              │
└─────────────┬───────────────────────────┘
              │
              ├─── Base Tenant: "acme"
              │    ├── Persistent Memory (ChromaDB/Upstash)
              │    │   ├── Brand knowledge
              │    │   ├── Product information
              │    │   └── Company policies
              │    └── File Storage
              │        └── /tenants/acme/files/
              │
              └─── Session: "+15551234567"
                   └── Ephemeral Memory (In-Memory)
                       ├── Conversation history
                       ├── User context
                       └── Temporary preferences
```

## Implementation Details

### 1. Core Components

#### Tenant/Session Parser
```python
def _parse_tenant_session(self, tenant: str) -> Tuple[str, Optional[str]]:
    """Parse tenant ID into base tenant and optional session ID"""
    if ':' in tenant:
        base_tenant, session_id = tenant.split(':', 1)
        return base_tenant, session_id
    return tenant, None
```

#### Session Memory Manager
```python
class SessionMemoryManager:
    def __init__(self, memory_model: str, max_recent: int = 20, summarize_after: int = 30):
        self.memory_model = memory_model
        self.max_recent = max_recent  # Keep last N messages raw
        self.summarize_after = summarize_after  # Trigger summary after N messages
        self.memories = []
        self.summaries = []  # Time-bucketed summaries
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
```

### 2. Memory Management Strategy (Hybrid Approach)

The system uses a three-tier memory structure to prevent overflow while maintaining context:

#### Tier 1: Recent Context (Last 20-30 messages)
- Stored in full detail
- Provides immediate conversation context
- Fast access for current interactions

#### Tier 2: Hourly Summaries (1-24 hours)
- Older messages compressed using `memory_model`
- Maintains key facts and decisions
- Reduces memory footprint by ~80%

#### Tier 3: Session Summary (Beyond 24 hours)
- High-level overview of entire session
- Key outcomes and decisions only
- Minimal memory usage

#### Summarization Flow
```
Messages 1-30   → Keep in full
Messages 31-50  → Summarize messages 1-20 into hourly bucket
Messages 51-70  → Summarize messages 21-40 into hourly bucket
...and so on
```

### 3. Session Lifecycle

#### Session Creation
- Created on first request with session ID
- Initialized with empty memory structures
- TTL set to `session_ttl_hours` (default: 24)

#### Session Access
- TTL refreshed on each access
- Memories retrieved and merged with base tenant
- Recent activity tracked

#### Session Expiration
- Checked periodically or on access
- Triggers `on_session_end` callback
- Cleans up memory structures

### 4. Configuration Options

```python
class BrainProxy:
    def __init__(
        self,
        # ... existing parameters ...
        
        # Session management
        enable_session_memory: bool = True,           # Feature flag
        session_ttl_hours: int = 24,                 # Session lifetime
        session_max_messages: int = 100,             # Hard limit per session
        session_summarize_after: int = 30,           # Trigger summarization
        session_memory_max_mb: float = 10.0,         # Memory limit per session
        on_session_end: Optional[Callable] = None,   # Cleanup callback
    ):
```

## Security Considerations

### 1. File Upload Blocking
Ephemeral sessions **cannot** upload files to prevent:
- Accidental pollution of base tenant knowledge
- Security vulnerabilities
- Data persistence confusion

```python
async def _ingest_files(self, files: List[FileData], tenant: str):
    base_tenant, session_id = self._parse_tenant_session(tenant)
    
    if session_id is not None:
        raise HTTPException(
            status_code=400,
            detail="File uploads not allowed for ephemeral sessions"
        )
```

### 2. Session ID Validation
- Pattern validation: `^[\w\+\-\.\@]+$`
- Length limits: max 128 characters
- Injection prevention through sanitization

### 3. Memory Limits
```python
SESSION_LIMITS = {
    "max_messages": 1000,          # Hard message limit
    "max_memory_mb": 10,           # Memory size limit
    "max_session_age_days": 7,     # Maximum age even with refresh
    "max_sessions_per_tenant": 100 # Prevent session flooding
}
```

## API Usage Examples

### Customer Support Chat
```python
# Phone support agent helping customer
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "I need help with order #12345"}
    ],
    api_base="http://localhost:8000/v1/acme:+15551234567"
)
```

### Email Support Thread
```python
# Email-based support maintaining context
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Following up on my return request"}
    ],
    api_base="http://localhost:8000/v1/acme:customer@email.com"
)
```

### Web Chat Session
```python
# Web chat with session ID
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What are your business hours?"}
    ],
    api_base="http://localhost:8000/v1/acme:web_session_abc123"
)
```

## Callbacks

### on_session_end Callback
Called when a session expires or is explicitly ended:

```python
async def on_session_end(full_tenant_id: str, session_data: Dict):
    """
    Extract valuable information from ended sessions
    
    Args:
        full_tenant_id: Complete tenant:session identifier
        session_data: {
            'messages': List[Dict],
            'summaries': List[Dict],
            'created_at': datetime,
            'last_accessed': datetime,
            'message_count': int
        }
    """
    base_tenant, session_id = full_tenant_id.split(':', 1)
    
    # Extract insights
    summary = await analyze_conversation(session_data)
    
    # Store valuable feedback
    if summary.has_product_feedback:
        await store_feedback(base_tenant, summary.feedback)
    
    # Analytics
    await log_interaction(session_id, summary)
```

## Memory Retrieval Strategy

When processing requests with session IDs:

1. **Parse tenant and session**
   ```python
   base_tenant, session_id = self._parse_tenant_session(tenant)
   ```

2. **Retrieve base tenant memories**
   - Search persistent vector store
   - Get brand knowledge, policies, etc.

3. **Retrieve session memories**
   - Get recent messages from session
   - Include relevant summaries

4. **Merge and prioritize**
   - Session context takes precedence for recent events
   - Base knowledge provides domain expertise

5. **Apply temporal filtering**
   - Use existing temporal awareness features
   - Filter by date ranges if specified

## Testing Scenarios

### 1. Basic Session Flow
- Create session with `tenant:session1`
- Send multiple messages
- Verify context maintained
- Check TTL refresh

### 2. Memory Overflow
- Send 100+ messages
- Verify summarization triggers
- Check memory limits enforced

### 3. Session Expiration
- Create session
- Wait for TTL expiration
- Verify cleanup and callback

### 4. File Upload Blocking
- Attempt file upload with session ID
- Verify rejection with appropriate error

### 5. Concurrent Sessions
- Multiple sessions for same base tenant
- Verify isolation between sessions
- Check base tenant memory sharing

## Migration & Compatibility

### Backward Compatibility
- Tenants without `:` work exactly as before
- No changes to existing API contracts
- Feature flag `enable_session_memory` for gradual rollout

### Future Enhancements
1. **Redis/Upstash Support** - For distributed deployments
2. **Session Transfer** - Move session between channels
3. **Session Analytics** - Built-in conversation metrics
4. **Custom TTL per Session** - Dynamic expiration based on use case
5. **Session Templates** - Pre-configured session types

## Performance Considerations

### Memory Usage
- Base overhead: ~1KB per session
- Average session: 10-50KB (with summaries)
- Maximum session: 10MB (configurable limit)

### CPU Impact
- Summarization: Async, non-blocking
- Cleanup: Lazy evaluation on access
- Search: Parallel base + session queries

### Scalability
- 1 instance: ~10,000 concurrent sessions
- Multiple instances: Requires Redis/Upstash
- Cleanup strategy: O(1) amortized

## Implementation Phases

### Phase 1: Core Functionality (Week 1)
- [x] Design document
- [ ] Tenant/session parsing
- [ ] In-memory session storage
- [ ] Basic TTL management
- [ ] File upload blocking

### Phase 2: Memory Management (Week 2)
- [ ] Hybrid summarization strategy
- [ ] Memory overflow prevention
- [ ] Session cleanup mechanism
- [ ] on_session_end callback

### Phase 3: Testing & Documentation (Week 3)
- [ ] Comprehensive test suite
- [ ] Performance benchmarks
- [ ] API documentation updates
- [ ] Example applications

### Phase 4: Advanced Features (Future)
- [ ] Redis/Upstash adapter
- [ ] Session analytics
- [ ] Custom session types
- [ ] Session transfer capabilities

## Conclusion

This ephemeral session memory feature provides a clean separation between persistent domain knowledge and temporary conversation context. It enables brain-proxy to handle multi-user scenarios effectively while maintaining security, performance, and backward compatibility.

The hybrid memory management approach ensures efficient resource usage while preserving conversation quality, making it ideal for customer support, interactive assistants, and multi-user applications.
