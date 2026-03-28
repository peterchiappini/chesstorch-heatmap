# Project: Chess AI Heatmap
An interactive web application that visualizes the "attention" of a chess AI engine using heatmaps overlaid on a digital chessboard. 

## Architecture
- **Frontend:** React, TypeScript, Vite. (To be deployed via Firebase Hosting).
- **Backend:** Python, FastAPI, PyTorch. (To be containerized/deployed separately). The Python backend is located in the /backend directory. Always use the virtual environment located at /backend/venv when running Python scripts or installing pip packages.
- **Communication:** The frontend sends FEN strings (board states) via HTTP POST to the backend, which returns an array of 64 float values (Grad-CAM weights) representing square attention.

## Frontend Stack & Libraries
- UI Component: `react-chessboard` 
- Game Logic: `chess.js` (for move validation and FEN generation)
- Visuals: Use the `customSquareStyles` prop on the chessboard to overlay colors dynamically based on the backend data.

## Commands
- **Install dependencies:** `npm install`
- **Run dev server:** `npm run dev`
- **Build:** `npm run build`

## Coding Guidelines
- Write clean, strongly-typed TypeScript.
- Favor functional components and React Hooks.
- Keep the frontend logic decoupled from the AI logic so we can easily swap out models later.