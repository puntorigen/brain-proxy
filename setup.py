from setuptools import setup, find_packages

setup(
    name="brain-proxy",
    version="0.1.77",
    description="OpenAI-compatible FastAPI router with Chroma + LangMem memory.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pablo Schaffner",
    author_email="pablo@puntorigen.com",
    url="https://github.com/puntorigen/brain-proxy",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1,<1.0.0",
        "openai>=1.68.2,<2.0.0",
        "langchain-chroma>=0.1.0",
        "langmem>=0.0.29",
        "tiktoken>=0.7.0",
        "pydantic>=2.5.0,<3.0.0",
        "langchain-openai>=0.1.0",
        "numpy>=1.24.0,<2.0.0",
        "litellm>=1.70.0",
        "langchain-litellm>=0.1.0",
        "dateparser>=1.1.0",
        "async-promptic>=5.0.0",
        "httpx>=0.24.0",
        "upstash-vector>=0.1.0"
    ],
    include_package_data=True,
)
