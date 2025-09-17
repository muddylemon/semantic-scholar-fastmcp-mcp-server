FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Add the working directory to the PYTHONPATH.
# This is the crucial fix that solves the ModuleNotFoundError.
ENV PYTHONPATH /app

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/jlowin/fastmcp.git
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose the port that the MCP server will run on
EXPOSE 8080


# Command to run the server using the refactored entry point
CMD ["python", "run.py"]