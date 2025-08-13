from setuptools import setup, find_packages

setup(
    name="brain-proxy",
    version="0.1.65",
    description="OpenAI-compatible FastAPI router with Chroma + LangMem memory.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pablo Schaffner",
    author_email="pablo@puntorigen.com",
    url="https://github.com/puntorigen/brain-proxy",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi",
        "openai",
        "langchain-chroma",
        "langmem",
        "tiktoken",
        "pydantic",
        "langchain-openai",
        "numpy<2.0.0",
        "litellm",
        "langchain_litellm",
        "dateparser",
        "async-promptic",
        "httpx",
        "upstash-vector"  # Add Upstash Vector dependency
    ],
    include_package_data=True,
)
