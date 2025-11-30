# AI Toolkit UI (Go + React)

This project is a port of the original AI Toolkit UI to a Golang backend with a React (Vite) frontend. It maintains feature parity with the original Next.js implementation while providing a lightweight, single-binary capable architecture.

## Architecture

- **Frontend**: React + TypeScript built with Vite
- **Backend**: Go (Golang) standard library + Prisma Client Go
- **Database**: SQLite (managed via Prisma)

## Prerequisites

- Go 1.21+
- Node.js 18+
- SQLite

## Getting Started

### 1. Setup Database

Initialize the Prisma client and generate the Go client code:

```bash
# Install Go dependencies
go mod tidy

# Generate Prisma Go client
go run github.com/steebchen/prisma-client-go generate

# Push database schema to SQLite file (creates aitk_db.db)
go run github.com/steebchen/prisma-client-go db push
```

### 2. Run Backend (Go)

Start the Go server which handles API requests and serves the frontend:

```bash
go run main.go
```
The server will start on `http://localhost:8080`.

### 3. Run Frontend (React/Vite)

For development with hot-reload, run the frontend separately:

```bash
npm install
npm run dev
```
The frontend dev server will start on `http://localhost:5173` (proxies API requests to localhost:8080).

### 4. Production Build

To build the frontend for production serving via the Go binary:

```bash
npm run build
```
This generates static files in `dist/`. The Go server is configured to serve these files when running in production mode (ensure you have file serving logic in your handlers if serving directly, or use Nginx/Reverse Proxy in front).

## Features

- **Jobs Management**: Create, edit, stop, and monitor training jobs.
- **Queue System**: Manage job queues with start/stop controls.
- **Dataset Management**: Upload images (drag & drop), captioning, and organization.
- **Monitoring**: Real-time CPU & GPU widgets.
- **Sample Viewer**: Advanced image viewer with keyboard navigation and controls.
- **Config Editor**: YAML-based job configuration with validation.

## Project Structure

- `/handlers`: Go API route handlers (Jobs, Queue, GPU, etc.)
- `/prisma`: Database schema and client generation
- `/src`: React frontend source code
  - `/components`: Reusable UI components
  - `/features`: Page-specific logic (Dashboard, Jobs, Datasets)
  - `/hooks`: Custom React hooks
- `main.go`: Entry point for the Go server
