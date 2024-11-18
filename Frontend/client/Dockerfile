# Use Node.js version 22.11.0 as the base image
FROM node:22.11.0

# Install 'serve' globally
RUN npm install -g serve

# Set working directory
WORKDIR /app

# Copy the build folder to the Docker container
COPY build/ /app/build/

# Expose port 3000 for the serve server
EXPOSE 3000

# Command to start the frontend server
CMD ["serve", "build"]
