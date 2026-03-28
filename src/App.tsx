import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { Chess, type Square } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import type { Arrow, PieceDropHandlerArgs } from 'react-chessboard';
import { useStockfish } from './useStockfish';
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
  if (!heatmap || heatmap.length < 64) return styles;
  for (let i = 0; i < 64; i++) {
    styles[SQUARES[i]] = {
      backgroundColor: `rgba(255, 0, 0, ${(heatmap[i] ?? 0) * 0.7})`,
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
  const [whitePersonality, setWhitePersonality] = useState('carlsen');
  const [blackPersonality, setBlackPersonality] = useState('tal');
  const [personalities, setPersonalities] = useState<Personality[]>([]);
  const [personalitySearch, setPersonalitySearch] = useState('');
  const [activeDropdown, setActiveDropdown] = useState<'white' | 'black' | null>(null);
  const [showTechnical, setShowTechnical] = useState(false);
  const [neuralDepth, setNeuralDepth] = useState<'layer1' | 'layer2' | 'layer3' | 'layer4'>('layer3');
  const [boardSize, setBoardSize] = useState(() => {
    if (typeof window !== 'undefined' && window.innerWidth <= 960) {
      return Math.floor(window.innerWidth - 32 - 24); // padding - eval bar
    }
    return 520;
  });
  const dropdownRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    const startX = e.clientX;
    const startSize = boardSize;

    const onMove = (ev: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = ev.clientX - startX;
      const newSize = Math.max(320, Math.min(800, startSize + delta));
      setBoardSize(newSize);
    };

    const onUp = () => {
      isDragging.current = false;
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [boardSize]);

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
    const stockfishEntry: Personality = {
      key: 'stockfish',
      label: 'Stockfish 18',
      player: 'Stockfish',
      time_control: 'Engine',
      source: 'wasm',
      games: 0,
      epochs: 0,
      accuracy: 0,
      loss: 0,
      positions: 0,
      description: 'Stockfish 18 running in your browser via WebAssembly. Searches ahead (unlike the CNN models) to find the objectively best move.',
    };
    fetch(`${API_URL}/api/personalities`)
      .then(r => r.json())
      .then(data => {
        const list = data.personalities?.length ? data.personalities : [];
        setPersonalities([stockfishEntry, ...list]);
      })
      .catch(() => setPersonalities([stockfishEntry]));
  }, []);

  // Derive active personality from whose turn it is
  const currentTurn = game.turn(); // 'w' or 'b'
  const personality = currentTurn === 'w' ? whitePersonality : blackPersonality;
  const whitePersonalityObj = personalities.find(p => p.key === whitePersonality);
  const blackPersonalityObj = personalities.find(p => p.key === blackPersonality);

  // Auto-size board on mobile resize/rotation
  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth <= 960 && !isDragging.current) {
        setBoardSize(Math.floor(window.innerWidth - 32 - 24));
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setActiveDropdown(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const isStockfish = personality === 'stockfish';

  /** Get the right personality for a given FEN (based on whose turn it is). */
  const personalityForFen = useCallback((fen: string) => {
    const side = fen.split(' ')[1];
    return side === 'b' ? blackPersonality : whitePersonality;
  }, [whitePersonality, blackPersonality]);

  /** Fetch heatmap if the personality for the given FEN is a CNN model, otherwise clear. */
  const fetchForFen = useCallback((fen: string) => {
    const p = personalityForFen(fen);
    if (p === 'stockfish') {
      setHeatmap(new Array(64).fill(0));
      setTopMoves([]);
      setPredictedArrow([]);
      return;
    }
    fetchHeatmap(fen, p, neuralDepth).then(applyHeatmapResponse);
  }, [personalityForFen, neuralDepth, applyHeatmapResponse]);

  // Fetch heatmap on mount and when personality or depth changes
  useEffect(() => {
    if (isStockfish) {
      // Clear CNN heatmap immediately, Stockfish arrow will appear when ready
      setHeatmap(new Array(64).fill(0));
      setTopMoves([]);
      setPredictedArrow([]);
      return;
    }
    const fen = currentMoveIndex > 0 ? replayToMove(moveHistory, currentMoveIndex) : game.fen();
    fetchForFen(fen);
  }, [personality, neuralDepth]);

  const filteredPersonalities = personalities.filter(p => {
    const q = personalitySearch.toLowerCase();
    return p.player.toLowerCase().includes(q)
      || p.time_control.toLowerCase().includes(q)
      || p.key.toLowerCase().includes(q);
  });
  const [pgnInput, setPgnInput] = useState('');
  const [pgnError, setPgnError] = useState('');
  const [isReplayMode, setIsReplayMode] = useState(false);
  const [boardOrientation, setBoardOrientation] = useState<'white' | 'black'>('white');

  const squareStyles = useMemo(() => buildSquareStyles(heatmap), [heatmap]);

  const currentFen = useMemo(
    () => replayToMove(moveHistory, currentMoveIndex),
    [moveHistory, currentMoveIndex],
  );

  const stockfish = useStockfish(currentFen);

  // When Stockfish is the active personality, show its best move as a green arrow
  useEffect(() => {
    if (!isStockfish) return;
    if (stockfish.bestMove && stockfish.bestMove.length >= 4) {
      const from = stockfish.bestMove.slice(0, 2);
      const to = stockfish.bestMove.slice(2, 4);
      setArrowFromMove([from, to] as [string, string], 'rgba(0, 180, 80, 0.7)');
    }
  }, [isStockfish, stockfish.bestMove]);

  const goToMove = useCallback(
    (index: number) => {
      const clamped = Math.max(0, Math.min(index, moveHistory.length));
      setCurrentMoveIndex(clamped);
      const fen = replayToMove(moveHistory, clamped);
      setGame(new Chess(fen));
      fetchForFen(fen);
    },
    [moveHistory, fetchForFen],
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
      fetchForFen(gameCopy.fen());
      return true;
    },
    [currentFen, moveHistory, currentMoveIndex, isReplayMode, fetchForFen],
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
      fetchForFen(gameCopy.fen());
    },
    [currentFen, moveHistory, currentMoveIndex, applyHeatmapResponse, fetchForFen],
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
    fetchForFen(startFen);
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
    fetchForFen(fresh.fen());
  }, [fetchForFen]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo-mark" />
        <h1>Chess AI <span>Heatmap</span></h1>
      </header>
      <main className="app-main">
        <div className="board-column" style={{ '--board-size': `${boardSize}px` } as React.CSSProperties}>
          <div className="personality-picker" ref={dropdownRef}>
            <div className="side-pickers">
              {(['white', 'black'] as const).map(side => {
                const sideKey = side === 'white' ? whitePersonality : blackPersonality;
                const setSide = side === 'white' ? setWhitePersonality : setBlackPersonality;
                const sideObj = side === 'white' ? whitePersonalityObj : blackPersonalityObj;
                const isOpen = activeDropdown === side;
                const isActive = (side === 'white' && currentTurn === 'w') || (side === 'black' && currentTurn === 'b');

                return (
                  <div key={side} className={`side-picker ${isActive ? 'side-active' : ''}`}>
                    <button
                      className={`personality-trigger side-${side}`}
                      onClick={() => setActiveDropdown(isOpen ? null : side)}
                    >
                      <span className={`side-badge side-badge-${side}`}>
                        {side === 'white' ? 'W' : 'B'}
                      </span>
                      <span className="personality-trigger-label">
                        {sideObj?.player ?? sideKey}
                      </span>
                      <span className="personality-trigger-arrow">{isOpen ? '\u25B2' : '\u25BC'}</span>
                    </button>
                    {isOpen && (
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
                              className={`personality-option${p.key === sideKey ? ' active' : ''}`}
                              onClick={() => {
                                setSide(p.key);
                                setActiveDropdown(null);
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
                  </div>
                );
              })}
            </div>
            <div className="side-stats-row">
              {([
                { side: 'white' as const, obj: whitePersonalityObj, turn: 'w' },
                { side: 'black' as const, obj: blackPersonalityObj, turn: 'b' },
              ]).map(({ side, obj, turn }) => (
                <div key={side} className={`side-stats ${currentTurn === turn ? 'side-stats-active' : ''}`}>
                  <span className={`side-stats-label`}>
                    <span className={`side-badge side-badge-${side} side-badge-sm`}>
                      {side === 'white' ? 'W' : 'B'}
                    </span>
                    {obj?.player ?? side}
                  </span>
                  {obj && obj.positions > 0 && (
                    <>
                      <div className="side-stats-detail">
                        <span>{obj.games.toLocaleString()} games</span>
                        <span className="stat-sep" />
                        <span className="stat-with-help">
                          {Math.round(obj.accuracy * 100)}% acc
                          <span className="help-icon">?<span className="help-tooltip">How often the model predicts the exact same move the GM played. Higher = better learned, but very high on small datasets may mean memorization.</span></span>
                        </span>
                        {obj.accuracy > 0.85 && obj.games < 500 && (
                          <span className="side-stats-warn" title="High accuracy on a small dataset usually means the model memorized specific positions rather than learning general patterns">memorized</span>
                        )}
                        {obj.accuracy > 0.85 && obj.games >= 500 && obj.games < 2000 && (
                          <span className="side-stats-warn" title="The model may be partially overfit — it learned some real patterns but also memorized common positions">overfit</span>
                        )}
                      </div>
                      {obj.description && (
                        <p className="side-stats-desc">{obj.description}</p>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
          <div className="board-with-eval">
            <div className="eval-bar">
              <div
                className="eval-bar-white"
                style={{ height: `${(() => {
                  if (stockfish.mate !== null) return stockfish.mate > 0 ? 100 : 0;
                  const clamped = Math.max(-500, Math.min(500, stockfish.score));
                  return 50 + (clamped / 500) * 50;
                })()}%` }}
              />
              <span className="eval-bar-label">
                {stockfish.mate !== null
                  ? `M${Math.abs(stockfish.mate)}`
                  : `${stockfish.score >= 0 ? '+' : ''}${(stockfish.score / 100).toFixed(1)}`
                }
              </span>
              <span className="eval-bar-depth">d{stockfish.depth}</span>
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

        <div className="resize-handle" onMouseDown={handleResizeStart} />

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
