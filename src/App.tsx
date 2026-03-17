import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { Chess, type Square } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import type { Arrow, PieceDropHandlerArgs } from 'react-chessboard';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SQUARES: Square[] = [
  'a8','b8','c8','d8','e8','f8','g8','h8',
  'a7','b7','c7','d7','e7','f7','g7','h7',
  'a6','b6','c6','d6','e6','f6','g6','h6',
  'a5','b5','c5','d5','e5','f5','g5','h5',
  'a4','b4','c4','d4','e4','f4','g4','h4',
  'a3','b3','c3','d3','e3','f3','g3','h3',
  'a2','b2','c2','d2','e2','f2','g2','h2',
  'a1','b1','c1','d1','e1','f1','g1','h1',
];

function generateRandomHeatmap(): number[] {
  return Array.from({ length: 64 }, () => Math.random());
}

interface TopMove {
  move: [string, string];
  confidence: number;
}

interface HeatmapResponse {
  weights: number[];
  top_moves?: TopMove[];
  game_over?: string;
  result?: string;
}

interface Personality {
  key: string;
  label: string;
  player: string;
  time_control: string;
  source: string;
  games: number;
  epochs: number;
  accuracy: number;
  loss: number;
  positions: number;
  description: string;
}

async function fetchHeatmap(fen: string, personality: string, depth = 'layer3'): Promise<HeatmapResponse> {
  try {
    const response = await fetch(`${API_URL}/api/heatmap`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, personality, depth }),
    });
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch heatmap from backend:', error);
    return { weights: generateRandomHeatmap() };
  }
}

const DEPTH_STEPS = [
  { key: 'layer1', label: 'Layer 1: Raw Vision', description: 'The AI is identifying where pieces are located — basic shapes and positions.' },
  { key: 'layer2', label: 'Layer 2: Tactical Patterns', description: 'The AI is recognizing tactical motifs — forks, pins, and piece coordination.' },
  { key: 'layer3', label: 'Layer 3: Complex Relationships', description: 'The AI is weighing strategic factors — pawn structure, king safety, and piece activity.' },
  { key: 'layer4', label: 'Layer 4: Global Strategy', description: 'The final layer combines everything into a holistic board assessment. Best with large datasets — may look noisy for models trained on few games.' },
] as const;

function buildSquareStyles(heatmap: number[]): Record<string, React.CSSProperties> {
  const styles: Record<string, React.CSSProperties> = {};
  for (let i = 0; i < 64; i++) {
    styles[SQUARES[i]] = {
      backgroundColor: `rgba(255, 0, 0, ${heatmap[i] * 0.7})`,
      transition: 'background-color 0.2s ease',
    };
  }
  return styles;
}

/** Replay a Chess game up to a specific move index and return the FEN. */
function replayToMove(moves: string[], moveIndex: number): string {
  const g = new Chess();
  for (let i = 0; i < moveIndex; i++) {
    g.move(moves[i]);
  }
  return g.fen();
}

function App() {
  const [game, setGame] = useState(new Chess());
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0);
  const [heatmap, setHeatmap] = useState<number[]>(generateRandomHeatmap);
  const [predictedArrow, setPredictedArrow] = useState<Arrow[]>([]);
  const [topMoves, setTopMoves] = useState<TopMove[]>([]);
  const [gameOver, setGameOver] = useState<string | null>(null);
  const [gameResult, setGameResult] = useState<string | null>(null);
  const [personality, setPersonality] = useState('carlsen');
  const [personalities, setPersonalities] = useState<Personality[]>([]);
  const [personalitySearch, setPersonalitySearch] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [showTechnical, setShowTechnical] = useState(false);
  const [neuralDepth, setNeuralDepth] = useState<'layer1' | 'layer2' | 'layer3' | 'layer4'>('layer3');
  const dropdownRef = useRef<HTMLDivElement>(null);

  const setArrowFromMove = useCallback((move: [string, string], color = 'rgba(0, 100, 255, 0.7)') => {
    setPredictedArrow([{ startSquare: move[0], endSquare: move[1], color }]);
  }, []);

  const applyHeatmapResponse = useCallback((data: HeatmapResponse) => {
    setHeatmap(data.weights);
    const moves = data.top_moves ?? [];
    setTopMoves(moves);
    if (moves.length > 0) {
      setArrowFromMove(moves[0].move);
    } else {
      setPredictedArrow([]);
    }
    setGameOver(data.game_over ?? null);
    setGameResult(data.result ?? null);
  }, [setArrowFromMove]);

  // Fetch available personalities from the API
  useEffect(() => {
    fetch(`${API_URL}/api/personalities`)
      .then(r => r.json())
      .then(data => {
        if (data.personalities?.length) setPersonalities(data.personalities);
      })
      .catch(() => {});
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Fetch heatmap on mount and when personality or depth changes
  useEffect(() => {
    const fen = currentMoveIndex > 0 ? replayToMove(moveHistory, currentMoveIndex) : game.fen();
    fetchHeatmap(fen, personality, neuralDepth).then(applyHeatmapResponse);
  }, [personality, neuralDepth]);

  const filteredPersonalities = personalities.filter(p => {
    const q = personalitySearch.toLowerCase();
    return p.player.toLowerCase().includes(q)
      || p.time_control.toLowerCase().includes(q)
      || p.key.toLowerCase().includes(q);
  });

  const selectedPersonality = personalities.find(p => p.key === personality);
  const [pgnInput, setPgnInput] = useState('');
  const [pgnError, setPgnError] = useState('');
  const [isReplayMode, setIsReplayMode] = useState(false);
  const [boardOrientation, setBoardOrientation] = useState<'white' | 'black'>('white');

  const squareStyles = useMemo(() => buildSquareStyles(heatmap), [heatmap]);

  const currentFen = useMemo(
    () => replayToMove(moveHistory, currentMoveIndex),
    [moveHistory, currentMoveIndex],
  );

  const goToMove = useCallback(
    (index: number) => {
      const clamped = Math.max(0, Math.min(index, moveHistory.length));
      setCurrentMoveIndex(clamped);
      const fen = replayToMove(moveHistory, clamped);
      setGame(new Chess(fen));
      fetchHeatmap(fen, personality, neuralDepth).then(applyHeatmapResponse);
    },
    [moveHistory, personality, neuralDepth],
  );

  const onPieceDrop = useCallback(
    ({ sourceSquare, targetSquare }: PieceDropHandlerArgs): boolean => {
      if (isReplayMode && currentMoveIndex < moveHistory.length) return false;

      const gameCopy = new Chess(currentFen);
      const move = gameCopy.move({
        from: sourceSquare,
        to: targetSquare ?? sourceSquare,
        promotion: 'q',
      });

      if (!move) return false;

      const newHistory = [...moveHistory.slice(0, currentMoveIndex), move.san];
      setMoveHistory(newHistory);
      setCurrentMoveIndex(newHistory.length);
      setGame(gameCopy);
      fetchHeatmap(gameCopy.fen(), personality, neuralDepth).then(applyHeatmapResponse);
      return true;
    },
    [currentFen, moveHistory, currentMoveIndex, isReplayMode, personality, neuralDepth],
  );

  const playMove = useCallback(
    (from: string, to: string) => {
      const gameCopy = new Chess(currentFen);
      const move = gameCopy.move({ from, to, promotion: 'q' });
      if (!move) return;

      const newHistory = [...moveHistory.slice(0, currentMoveIndex), move.san];
      setMoveHistory(newHistory);
      setCurrentMoveIndex(newHistory.length);
      setGame(gameCopy);
      fetchHeatmap(gameCopy.fen(), personality, neuralDepth).then(applyHeatmapResponse);
    },
    [currentFen, moveHistory, currentMoveIndex, applyHeatmapResponse, personality, neuralDepth],
  );

  const handleLoadPgn = useCallback(() => {
    setPgnError('');
    const loader = new Chess();
    try {
      loader.loadPgn(pgnInput);
    } catch {
      setPgnError('Invalid PGN — check the format and try again.');
      return;
    }

    const history = loader.history();
    if (history.length === 0) {
      setPgnError('No moves found in this PGN.');
      return;
    }

    setMoveHistory(history);
    setCurrentMoveIndex(0);
    setIsReplayMode(true);
    const startFen = new Chess().fen();
    setGame(new Chess(startFen));
    fetchHeatmap(startFen, personality, neuralDepth).then(applyHeatmapResponse);
  }, [pgnInput, personality, neuralDepth]);

  const handleReset = useCallback(() => {
    const fresh = new Chess();
    setGame(fresh);
    setMoveHistory([]);
    setCurrentMoveIndex(0);
    setIsReplayMode(false);
    setPgnInput('');
    setPgnError('');
    setGameOver(null);
    setGameResult(null);
    setTopMoves([]);
    fetchHeatmap(fresh.fen(), personality, neuralDepth).then(applyHeatmapResponse);
  }, [personality, neuralDepth]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo-mark" />
        <h1>Chess AI <span>Heatmap</span></h1>
      </header>
      <main className="app-main">
        <div className="board-column">
          <div className="personality-picker" ref={dropdownRef}>
            <button
              className="personality-trigger"
              onClick={() => setDropdownOpen(o => !o)}
            >
              <span className="personality-trigger-label">
                {selectedPersonality?.label ?? personality}
              </span>
              <span className="personality-trigger-arrow">{dropdownOpen ? '\u25B2' : '\u25BC'}</span>
            </button>
            {dropdownOpen && (
              <div className="personality-dropdown">
                <input
                  className="personality-search"
                  type="text"
                  placeholder="Search players..."
                  value={personalitySearch}
                  onChange={e => setPersonalitySearch(e.target.value)}
                  autoFocus
                />
                <ul className="personality-list">
                  {filteredPersonalities.length === 0 && (
                    <li className="personality-empty">No matches</li>
                  )}
                  {filteredPersonalities.map(p => (
                    <li
                      key={p.key}
                      className={`personality-option${p.key === personality ? ' active' : ''}`}
                      onClick={() => {
                        setPersonality(p.key);
                        setDropdownOpen(false);
                        setPersonalitySearch('');
                      }}
                    >
                      <span className="personality-option-name">{p.player}</span>
                      {p.time_control && (
                        <span className={`personality-option-tc tc-${p.time_control.toLowerCase()}`}>
                          {p.time_control}
                        </span>
                      )}
                      {p.accuracy > 0 && (
                        <span className="personality-option-acc">
                          {Math.round(p.accuracy * 100)}%
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {selectedPersonality && selectedPersonality.positions > 0 && (
              <>
                <div className="personality-stats">
                  <span>{selectedPersonality.games.toLocaleString()} games</span>
                  <span className="stat-sep" />
                  <span>{selectedPersonality.positions.toLocaleString()} positions</span>
                  <span className="stat-sep" />
                  <span>{selectedPersonality.epochs} epochs</span>
                  <span className="stat-sep" />
                  <span>{Math.round(selectedPersonality.accuracy * 100)}% acc</span>
                  <span className="stat-sep" />
                  <span>loss {selectedPersonality.loss}</span>
                </div>
                {selectedPersonality.games > 0 && selectedPersonality.games < 500 && (
                  <p className="overfit-warning">
                    Small dataset — model may be overfit. Deeper layers may appear noisy.
                  </p>
                )}
                {selectedPersonality.description && (
                  <p className="personality-description">
                    {selectedPersonality.description}
                  </p>
                )}
              </>
            )}
          </div>
          <div className="board-container">
            <Chessboard
              options={{
                position: currentFen,
                onPieceDrop,
                squareStyles,
                arrows: predictedArrow,
                boardOrientation,
                allowDragging: !(isReplayMode && currentMoveIndex < moveHistory.length),
              }}
            />
          </div>

          <div className="controls">
            <button
              className="control-btn"
              disabled={currentMoveIndex === 0}
              onClick={() => goToMove(0)}
              title="Go to start"
            >
              &#x23EE;
            </button>
            <button
              className="control-btn"
              disabled={currentMoveIndex === 0}
              onClick={() => goToMove(currentMoveIndex - 1)}
            >
              &#x25C0; Prev
            </button>
            <span className="move-counter">
              {currentMoveIndex} / {moveHistory.length}
            </span>
            <button
              className="control-btn"
              disabled={currentMoveIndex >= moveHistory.length}
              onClick={() => goToMove(currentMoveIndex + 1)}
            >
              Next &#x25B6;
            </button>
            <button
              className="control-btn"
              disabled={currentMoveIndex >= moveHistory.length}
              onClick={() => goToMove(moveHistory.length)}
              title="Go to end"
            >
              &#x23ED;
            </button>
            <button
              className="control-btn flip-btn"
              onClick={() => setBoardOrientation(o => o === 'white' ? 'black' : 'white')}
              title="Flip board"
            >
              &#x21C5;
            </button>
            <button className="control-btn reset-btn" onClick={handleReset}>
              Reset
            </button>
          </div>

          <div className="depth-switcher">
            {DEPTH_STEPS.map((step) => (
              <button
                key={step.key}
                className={`depth-btn${neuralDepth === step.key ? ' active' : ''}`}
                onClick={() => setNeuralDepth(step.key as typeof neuralDepth)}
                title={step.description}
              >
                {step.label}
              </button>
            ))}
          </div>
          <p className="depth-description">
            {DEPTH_STEPS.find(s => s.key === neuralDepth)?.description}
          </p>

          {moveHistory.length > 0 && (
            <div className="move-list">
              {moveHistory.map((san, i) => {
                const isWhite = i % 2 === 0;
                return (
                  <span key={i}>
                    {isWhite && (
                      <span className="move-number">{Math.floor(i / 2) + 1}.</span>
                    )}
                    <button
                      className={`move-san${i + 1 === currentMoveIndex ? ' active' : ''}`}
                      onClick={() => goToMove(i + 1)}
                    >
                      {san}
                    </button>
                  </span>
                );
              })}
              {gameResult && (
                <span className="game-result" title={gameOver ?? ''}>{gameResult}</span>
              )}
            </div>
          )}

          <div className="pgn-loader">
            <textarea
              className="pgn-input"
              placeholder={'Paste PGN here, e.g.\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 ...'}
              value={pgnInput}
              onChange={(e) => {
                setPgnInput(e.target.value);
                setPgnError('');
              }}
              rows={3}
            />
            {pgnError && <p className="pgn-error">{pgnError}</p>}
            <button
              className="control-btn load-pgn-btn"
              onClick={handleLoadPgn}
              disabled={pgnInput.trim().length === 0}
            >
              Load PGN
            </button>
          </div>
        </div>

        <aside className="sidebar">
          <div className="sidebar-inner">
            {topMoves.length > 0 && (
              <>
                <div className="sidebar-section ai-thoughts">
                  <h2>AI Thoughts</h2>
                  <ul className="top-moves-list">
                    {topMoves.map((tm, i) => (
                      <li
                        key={i}
                        className="top-move-item"
                        onClick={() => playMove(tm.move[0], tm.move[1])}
                        onMouseEnter={() => setArrowFromMove(tm.move, i === 0
                          ? 'rgba(0, 100, 255, 0.7)'
                          : 'rgba(100, 100, 255, 0.45)'
                        )}
                        onMouseLeave={() => {
                          if (topMoves.length > 0) setArrowFromMove(topMoves[0].move);
                        }}
                      >
                        <span className="top-move-rank">#{i + 1}</span>
                        <span className="top-move-label">
                          {tm.move[0]} → {tm.move[1]}
                        </span>
                        <div className="confidence-bar-track">
                          <div
                            className="confidence-bar-fill"
                            style={{ width: `${Math.round(tm.confidence * 100)}%` }}
                          />
                        </div>
                        <span className="top-move-pct">
                          {Math.round(tm.confidence * 100)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
                <hr className="sidebar-divider" />
              </>
            )}

            <div className="sidebar-section">
              <h2>What is this heatmap?</h2>
              <p>
                Unlike a traditional engine that just tells you the "best" move,
                this AI shows you its <strong>attention</strong> — imagine it has
                a spotlight it can shine across the board. The brighter the red
                glow on a square, the harder the AI is staring at it to make its
                decision.
              </p>
              <ul className="heatmap-legend">
                <li>
                  <span className="legend-swatch legend-bright" />
                  <span><strong>Bright red</strong> — high influence. These squares
                  most shaped the AI's chosen move.</span>
                </li>
                <li>
                  <span className="legend-swatch legend-faint" />
                  <span><strong>Faint tint</strong> — low influence. The AI noticed
                  these squares but didn't weigh them heavily.</span>
                </li>
                <li>
                  <span className="legend-swatch legend-arrow" />
                  <span><strong>Blue arrow</strong> — the move the AI would play
                  from this position.</span>
                </li>
              </ul>
            </div>

            <hr className="sidebar-divider" />

            <div className="sidebar-section">
              <h2>Inside the Brain</h2>

              {selectedPersonality && selectedPersonality.games > 0 && (
                <div className="training-badge">
                  Trained on {selectedPersonality.games.toLocaleString()} {selectedPersonality.label} games
                </div>
              )}

              <ul className="brain-explainer">
                <li>
                  <span className="brain-icon">👁</span>
                  <div>
                    <strong>Pattern Recognition (The CNN)</strong>
                    <p>
                      This AI doesn't "calculate" like a calculator. It uses
                      a <em>Convolutional Neural Network</em> — the same
                      architecture that powers image recognition. It slides small
                      3x3 "filters" across the board, learning to spot shapes
                      like open files, pawn chains, and piece clusters, the way
                      your eye recognizes a face without counting pixels.
                    </p>
                  </div>
                </li>
                <li>
                  <span className="brain-icon">⚡</span>
                  <div>
                    <strong>Pure Instinct (No Look-Ahead)</strong>
                    <p>
                      Unlike Stockfish, this model doesn't search ahead. It sees
                      the current board and reacts instantly — playing on gut
                      feeling learned from Grandmaster games. That's why it
                      might suggest a brilliant positional move one second, then
                      miss a simple tactic the next.
                    </p>
                  </div>
                </li>
                <li>
                  <span className="brain-icon">🔴</span>
                  <div>
                    <strong>The Heatmap (Grad-CAM)</strong>
                    <p>
                      The red glow isn't where the piece is going — it's what
                      the AI is <em>staring at</em>. We trace the decision
                      backwards through the network's layers to find which
                      squares triggered its neurons the most. The brighter the
                      red, the more that square influenced the decision.
                    </p>
                  </div>
                </li>
              </ul>
            </div>

            <hr className="sidebar-divider" />

            <div className="sidebar-section">
              <h2>Powered by PyTorch</h2>
              <p>
                This model is built with <strong>PyTorch</strong>, the open-source
                framework used by researchers at Meta, Tesla, and OpenAI. Here's
                the pipeline that runs every time you move a piece:
              </p>
              <ol className="pytorch-pipeline">
                <li>
                  <strong>Encode</strong> — The board becomes a 17-layer
                  image: 12 layers for piece positions (one per piece type and
                  color), 4 for castling rights, and 1 for whose turn it is.
                </li>
                <li>
                  <strong>Forward pass</strong> — PyTorch pushes this
                  "image" through 4 convolutional blocks (64 → 128 → 256 → 256
                  filters), each one detecting increasingly complex patterns.
                </li>
                <li>
                  <strong>Two-headed prediction</strong> — The network
                  outputs two sets of 64 scores: one for "which piece to move"
                  and one for "where to move it." Legal move masking ensures
                  only valid moves are considered.
                </li>
                <li>
                  <strong>Grad-CAM</strong> — We run the decision
                  backwards through the network, asking "which squares mattered
                  most?" The gradients reveal the answer — painted as the
                  heatmap you see.
                </li>
              </ol>
            </div>

            <hr className="sidebar-divider" />

            <div className="sidebar-section">
              <button
                className="technical-toggle"
                onClick={() => setShowTechnical(prev => !prev)}
              >
                {showTechnical ? 'Hide' : 'Show'} Technical Details
              </button>
              {showTechnical && (
                <div className="technical-details">
                  <p>
                    <strong>Architecture:</strong> 4-block CNN. Each block
                    is Conv2d(3x3, padding=1) → BatchNorm → ReLU. All layers
                    maintain 8x8 spatial resolution (no pooling), giving a 1:1
                    mapping between feature-map cells and board squares.
                  </p>
                  <p>
                    <strong>Input tensor:</strong> (17, 8, 8) — 12 piece-type
                    planes (6 white + 6 black), 4 castling planes, 1
                    side-to-move plane. Each plane is binary (0 or 1).
                  </p>
                  <p>
                    <strong>Output:</strong> Two linear heads (fc_from, fc_to),
                    each projecting 256x8x8 = 16,384 features to 64 logits.
                    Trained with CrossEntropyLoss on both heads summed.
                  </p>
                  <p>
                    <strong>Grad-CAM:</strong> Forward + backward hooks on each
                    conv block capture activations and gradients. Channel weights
                    are global-average-pooled gradients. CAM = ReLU(weighted sum
                    of feature maps). Contrast thresholding zeros below-mean
                    values, then squares the rest for visual pop.
                  </p>
                  <p>
                    <strong>Legal masking:</strong> From-head logits are masked
                    to legal origin squares (-1e9 on illegal), then to-head is
                    masked to legal destinations from the chosen piece.
                  </p>
                </div>
              )}
            </div>

            <p className="sidebar-hint">
              Move a piece and watch the heatmap shift — you're watching a
              neural network change its mind.
            </p>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
