#!/bin/bash

echo "🚀 Starting NeedsMatcher..."

# Install all dependencies
echo "📦 Installing dependencies..."
cd backend
npm run install-deps
cd ..

# Start both services using npm
echo "⚡ Starting services..."
cd backend
npm run dev 