# ----------------------------------------------------------------------
# Stage 1: Build - Install Dependencies and Copy Source
FROM python:3.11-slim-bookworm AS builder

# Set the working directory inside the container
WORKDIR /app

# Copy dependency file first to leverage Docker's build cache
COPY requirements.txt .

# Install dependencies (streamlit is installed here, fixing the 'executable not found' error)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the essential application files needed at runtime
COPY src ./src
COPY streamlit_app ./streamlit_app
COPY data ./data 
COPY .env ./.env 
# FIX: Re-adding models directory copy to resolve 'Model file not found' error
COPY models ./models 

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Stage 2: Production/Runtime Image
# Use the 'builder' stage as the final image, leveraging all previous setup
FROM builder AS final

# Fix the warning: use key=value format for ENV
ENV PYTHONUNBUFFERED=1

# Streamlit defaults to port 8501
EXPOSE 8501

# Define the command to start the Streamlit application
CMD [ "streamlit", "run", "streamlit_app/app.py", "--server.port", "8501", "--server.enableCORS", "False" ]