# Give Your FastAPI App a Brain: Introducing **Brain-Proxy** for AI Memory & Multi‑Tenant Magic

*What if your AI never forgot a conversation?* Imagine an app that remembers every user interaction, provides instant access to relevant knowledge, and scales those smarts across many users — without rewriting a single line of your OpenAI client code. That’s what **`brain-proxy`** offers. While memory and RAG are becoming common in modern LLM apps, what sets brain-proxy apart is that it makes these features available **to any OpenAI-compatible client without modifications**. In this article, we’ll explore how `brain-proxy` can turn any FastAPI backend into a memory-augmented, RAG-capable, multi-tenant AI proxy. We’ll walk through real examples, common pain points, and show how brain-proxy solves them — all with friendly code and imaginative storytelling. Let’s dive in!

## 🚀 What is Brain-Proxy (and Why Should You Care)?

**Brain-Proxy** is a new Python package that supercharges your FastAPI app with AI capabilities: **multi-tenant memory, context retention, RAG (Retrieval-Augmented Generation), and file ingestion**, all behind an OpenAI-compatible API. In plain terms, it means you can build your own “ChatGPT-like” service that:

- **Remembers context** for each user (like how ChatGPT’s UI recalls earlier messages).
- **Keeps conversations separate** for different users or agents (multi-tenant by design).
- **Augments responses with external knowledge** via RAG – *pulling in facts from documents or databases on the fly*.
- **Lets you feed in files or data** as part of the AI’s knowledge (imagine uploading a PDF and then asking questions about it).

All of this comes without you writing complex state management or vector database code. **In fact, OpenAI’s own API doesn’t maintain conversational memory for you** – you normally have to send the entire conversation each time. Brain-Proxy changes that: it gives your app “long-term memory” out of the box. 

> **Takeaway:** *Brain-Proxy adds a “brain” to your backend – your app will not only think, but also remember.* 💡

## 📝 Example 1: An AI Diary with Eternal Memory

Let’s start with a simple, relatable scenario. Imagine building a personal **AI Diary**. Each user can “write” to their diary (via an AI) and later ask questions or get summaries of past entries. We want the diary AI to **always remember previous entries** – without us manually storing and retrieving chat history.

**With brain-proxy, eternal memory comes standard.** By integrating it into a FastAPI app, each user’s conversations are automatically stored and retrieved on subsequent requests. There’s no extra config or database plumbing needed on your part; just use the OpenAI API format and the proxy does the rest.

Here’s how we might set up a basic FastAPI app with brain-proxy:

```python
from fastapi import FastAPI
from brain_proxy import BrainProxy

app = FastAPI()
brain = BrainProxy()  # initialize the brain-proxy with default settings (memory enabled)
app.include_router(brain.router, prefix="/v1")  # expose OpenAI-compatible endpoints under /v1
```

With those few lines, our FastAPI server now speaks the OpenAI protocol. We can use the regular OpenAI client SDK to talk to it, but now **the AI has memory**. For example, let’s simulate a user writing to their diary and then asking a question:

```python
import openai
openai.api_key = "NOT-NEEDED"  # brain-proxy can accept a dummy key for local use
openai.api_base = "http://localhost:8000/v1/alice"  # the final part identifies the tenant or user

# User 'alice' writes a diary entry
response1 = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Dear Diary, I built my first FastAPI app today!"}]
)
print(response1.choices[0].message.content)

# Later, Alice asks her AI diary a question.
response2 = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What did I accomplish today?"}]
)
print(response2.choices[0].message.content)
```

In the above code, notice how we configure the `api_base` to include the user or tenant ID (`/v1/alice`). That identifier is key: **brain-proxy uses it to keep track of Alice’s conversation history.** The first request logs Alice’s entry; the second request builds on her earlier inputs — even without explicitly passing them again. Thanks to the new `temporal_awareness` feature introduced in v0.0.6, the AI can now intelligently filter memories by time, making it possible to ask things like “What did I write last week?” or “What’s planned for next month?” and get accurate, context-aware responses.

All of this happened with essentially zero effort on our side – we didn’t write a database or session handler. Brain-Proxy handled the heavy lifting of storing and retrieving Alice’s chat history (our “eternal memory”). **Every user can have their own persistent diary like this, concurrently, simply by using their own custom tenant endpoint**. For example, `/v1/bob`, `/v1/carol`, or `/v1/alice` – each of these acts as a separate memory space. That’s the *multi-tenant memory*: one service, many personal brains.

---

## 🤯 The Challenge: Why Is This Normally So Hard?

If you've ever tried building an AI app with context, you know the struggle:

- 🧠 **Memory management is hard** — You have to manually save, retrieve, and trim previous messages.
- 🧑‍🤝‍🧑 **User separation is critical** — It's too easy to accidentally mix user data.
- 📎 **RAG setup is complex** — Document ingestion, embedding, and search pipelines can be fragile.
- 🔐 **Security, usage tracking, and customization hooks** — These often require boilerplate or hacks.

That's a lot of moving parts, even for experienced teams.

## ⏰ New in 0.0.6: Temporal Awareness

One of the most powerful new features in brain-proxy 0.0.6 is **temporal awareness**. Now, when users ask time-based questions like:

- *“What did I do last week?”*
- *“What are my plans for next month?”*
- *“What happened on March 10th?”*

…the system understands those time references and **filters memory accordingly**, retrieving only the relevant context.

This works out of the box thanks to:

- Built-in time range detection
- Short- and long-term memory filtering by timestamp
- Seamless integration with the memory pipeline

This makes the assistant feel more natural and intelligent — especially in long-running or periodic use cases.

Enable it with a single flag:

```python
proxy = BrainProxy(
    default_model="openai/gpt-4o-mini",
    enable_memory=True,
    temporal_awareness=True  # 🔥 This enables time-based filtering
)
```

You can now build:

- ✅ Productivity assistants that summarize daily/weekly logs
- ✅ Goal tracking bots that recall future and past plans
- ✅ Event-aware agents for journaling or CRM history

---

## 🧠 How Brain-Proxy Fixes It

Brain-Proxy bakes all of this into one neat package:

- ✅ Auto-memory: short- and long-term memory for each tenant
- ✅ Tenant-aware routing: `/v1/<tenant>/chat/completions`
- ✅ Built-in RAG: via ChromaDB and LangChain
- ✅ File ingestion: just upload a `file_data` message
- ✅ Hooks: `auth_hook`, `usage_hook`, `manager_fn`

Now let’s level up the example.

## 🤖 Example 2: Multi-Agent Systems with LangGraph and Brain-Proxy

Let’s take a more structured, production-ready approach: **using LangGraph to coordinate multiple agents**, each powered by its own persistent memory and tenant context via brain-proxy.

In this setup:

- Each LangGraph node is an AI agent (PM, Engineer, Designer, etc.)
- Each node sends requests to a unique `brain-proxy` endpoint (`/v1/pm`, `/v1/engineer`, etc.)
- Agents maintain their own memory across sessions
- LangGraph orchestrates message passing, task flow, and coordination

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_agent
from openai import OpenAI

# Define a LangGraph state machine with three roles
roles = ["pm", "engineer", "designer"]

# Each node uses a different tenant-specific client
clients = {
    role: OpenAI(base_url=f"http://localhost:8000/v1/{role}", api_key="fake")
    for role in roles
}

# Define an agent node for each role
nodes = {
    role: tools_agent(clients[role], name=role.title())
    for role in roles
}

# Set up the LangGraph
graph = StateGraph()

# Connect the roles in a loop for collaborative brainstorming
for i, role in enumerate(roles):
    next_role = roles[(i + 1) % len(roles)]
    graph.add_node(role, nodes[role])
    graph.add_edge(role, next_role)

graph.set_entry_point("pm")
graph.set_finish_point("pm")  # Stops when PM receives the final response
workflow = graph.compile()

# Run the collaborative agent workflow
output = workflow.invoke({
    "messages": [{"role": "user", "content": "How can we improve onboarding?"}]
})

print(output["messages"][-1]["content"])
```

This example shows how brain-proxy fits naturally into orchestrated multi-agent flows. Each agent speaks from its own long-term memory, while LangGraph ensures structure and repeatability. You can scale this up with file ingestion, RAG, agent roles, and cross-agent reasoning.

**Bonus**: You can even inject `manager_fn` hooks in brain-proxy to simulate or augment decision-making between LangGraph turns.

## 💬 Final Thoughts

If you're building an AI product — assistant, agent, chatbot, internal tool — **brain-proxy helps you skip the glue code** and focus on building real value.

- Want persistent user memory? ✅
- Want multi-agent separation? ✅
- Want easy file ingestion + RAG? ✅

Give it a try:

```bash
pip install brain-proxy
```

Whether you’re a solo hacker or leading a team — **your backend deserves a brain.**

> What could your AI app do if it could remember, retrieve, and reason more like a human?

Let’s find out. 🚀

