/**
 * Global Forest Analyzer - Electron Main Process
 * ===============================================
 * 
 * FIXED for M-chip Macs with better Python detection
 * 
 * Features:
 * - Intelligent Python detection (Homebrew, system, conda)
 * - Chunked binary file reading for large datasets
 * - Visualization data loading
 * - Analysis integration via Python subprocess
 * - Region management
 */

const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');

// ═══════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════

const DATA_DIR = path.join(__dirname, 'deckgl_data');
const TILES_DIR = path.join(__dirname, 'output', 'tiles');
const ANALYSIS_SCRIPT = path.join(__dirname, 'analysis', 'analysis_cli.py');
const REGIONS_FILE = path.join(__dirname, 'analysis', 'regions.json');
const TEMP_DIR = path.join(__dirname, 'temp_analysis');

// Binary format: 20 bytes per point
const BYTES_PER_POINT = 20;

// Chunk size for reading large files (100MB chunks)
const READ_CHUNK_SIZE = 100 * 1024 * 1024;

// Maximum points to transfer for visualization (prevents IPC overflow)
const MAX_VISUALIZATION_POINTS = 20000000;

let mainWindow;
let REGIONS = {};
let PYTHON_PATH = null;

// ═══════════════════════════════════════════════════════════════
// PYTHON DETECTION - CRITICAL FIX FOR M-CHIP MACS
// ═══════════════════════════════════════════════════════════════

/**
 * Find Python executable on the system
 * Checks multiple common locations, especially for M-chip Macs
 */
function findPython() {
  const possiblePaths = [
    // Homebrew on M-chip Macs
    '/opt/homebrew/bin/python3',
    '/opt/homebrew/bin/python',
    
    // Homebrew on Intel Macs
    '/usr/local/bin/python3',
    '/usr/local/bin/python',
    
    // System Python (macOS)
    '/usr/bin/python3',
    '/usr/bin/python',
    
    // Conda/Anaconda
    path.join(process.env.HOME || '', 'anaconda3', 'bin', 'python'),
    path.join(process.env.HOME || '', 'miniconda3', 'bin', 'python'),
    
    // Generic PATH search
    'python3',
    'python'
  ];
  
  console.log('[Python Detection] Searching for Python...');
  
  for (const pythonPath of possiblePaths) {
    try {
      // Check if path exists (for absolute paths)
      if (path.isAbsolute(pythonPath) && !fs.existsSync(pythonPath)) {
        continue;
      }
      
      // Try to run python --version
      const version = execSync(`"${pythonPath}" --version`, {
        encoding: 'utf-8',
        timeout: 5000,
        stdio: ['ignore', 'pipe', 'ignore']
      }).trim();
      
      console.log(`[Python Detection] Found: ${pythonPath}`);
      console.log(`[Python Detection] Version: ${version}`);
      
      // Verify it has required packages
      try {
        execSync(`"${pythonPath}" -c "import numpy, pandas, scipy"`, {
          timeout: 5000,
          stdio: 'ignore'
        });
        console.log('[Python Detection] ✓ Required packages found (numpy, pandas, scipy)');
        return pythonPath;
      } catch (e) {
        console.log(`[Python Detection] ✗ ${pythonPath} missing required packages`);
      }
      
    } catch (e) {
      // This Python path doesn't work, try next
      continue;
    }
  }
  
  return null;
}

/**
 * Test Python installation and show helpful error if not found
 */
async function verifyPython() {
  PYTHON_PATH = findPython();
  
  if (!PYTHON_PATH) {
    console.error('[Python] ✗ Python not found!');
    
    // Show error dialog to user
    const result = await dialog.showMessageBox({
      type: 'error',
      title: 'Python Not Found',
      message: 'Python 3 is required but was not found on your system.',
      detail: 
        'The Forest Analyzer needs Python 3 with numpy, pandas, and scipy.\n\n' +
        'To install:\n' +
        '1. Install Homebrew: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n' +
        '2. Install Python: brew install python\n' +
        '3. Install packages: pip3 install numpy pandas scipy\n\n' +
        'After installation, restart this app.',
      buttons: ['Quit', 'Continue Without Analysis']
    });
    
    if (result.response === 0) {
      app.quit();
      return false;
    }
  } else {
    console.log(`[Python] ✓ Using: ${PYTHON_PATH}`);
  }
  
  return true;
}

// ═══════════════════════════════════════════════════════════════
// REGIONS LOADING
// ═══════════════════════════════════════════════════════════════

function loadRegions() {
  try {
    if (fs.existsSync(REGIONS_FILE)) {
      REGIONS = JSON.parse(fs.readFileSync(REGIONS_FILE, 'utf-8'));
      console.log(`[Regions] Loaded ${Object.keys(REGIONS).length} regions`);
    } else {
      console.log('[Regions] No regions.json found, using defaults');
      REGIONS = getDefaultRegions();
    }
  } catch (e) {
    console.error('[Regions] Failed to load:', e.message);
    REGIONS = getDefaultRegions();
  }
}

function getDefaultRegions() {
  return {
    amazon_west: {
      name: "Western Amazon Basin",
      bounds: { minLat: -15, maxLat: 5, minLon: -80, maxLon: -60 },
      biome: "tropical_moist",
      countries: ["Peru", "Ecuador", "Colombia", "Brazil"]
    },
    amazon_east: {
      name: "Eastern Amazon",
      bounds: { minLat: -15, maxLat: 0, minLon: -60, maxLon: -45 },
      biome: "tropical_moist",
      countries: ["Brazil"]
    },
    congo_basin: {
      name: "Congo Basin",
      bounds: { minLat: -10, maxLat: 8, minLon: 10, maxLon: 32 },
      biome: "tropical_moist",
      countries: ["DRC", "Republic of Congo", "Gabon", "Cameroon"]
    }
  };
}

// ═══════════════════════════════════════════════════════════════
// WINDOW SETUP
// ═══════════════════════════════════════════════════════════════

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1200,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    backgroundColor: '#0a0a1a',
    title: 'Global Forest Analyzer'
  });

  mainWindow.loadFile('viewer.html');

  // Create temp directory
  if (!fs.existsSync(TEMP_DIR)) {
    fs.mkdirSync(TEMP_DIR, { recursive: true });
  }

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
}

app.commandLine.appendSwitch('js-flags', '--max-old-space-size=16384');

app.whenReady().then(async () => {
  await verifyPython();
  loadRegions();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// ═══════════════════════════════════════════════════════════════
// LARGE FILE READING UTILITIES
// ═══════════════════════════════════════════════════════════════

function downsampleForVisualization(data, maxPoints) {
  const { totalPoints, totalPopulation, lats, lons, canopy, popDensityLog, lossYear, popCount, lossFrac } = data;
  
  if (totalPoints <= maxPoints) {
    console.log(`[Data] No downsampling needed (${totalPoints.toLocaleString()} <= ${maxPoints.toLocaleString()})`);
    return data;
  }
  
  const sampleRate = maxPoints / totalPoints;
  console.log(`[Data] Downsampling: ${totalPoints.toLocaleString()} -> ~${maxPoints.toLocaleString()} points (${(sampleRate * 100).toFixed(1)}%)`);
  
  const gridSize = 0.1;
  const grid = new Map();
  
  for (let i = 0; i < totalPoints; i++) {
    const gridKey = `${Math.floor(lats[i] / gridSize)},${Math.floor(lons[i] / gridSize)}`;
    if (!grid.has(gridKey)) {
      grid.set(gridKey, []);
    }
    grid.get(gridKey).push(i);
  }
  
  console.log(`[Data] Grid cells: ${grid.size.toLocaleString()}`);
  
  const sampledIndices = [];
  const pointsPerCell = Math.max(1, Math.floor(maxPoints / grid.size));
  
  for (const [key, indices] of grid) {
    const cellSampleCount = Math.min(indices.length, pointsPerCell);
    
    if (indices.length <= cellSampleCount) {
      sampledIndices.push(...indices);
    } else {
      const shuffled = indices.slice();
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      sampledIndices.push(...shuffled.slice(0, cellSampleCount));
    }
    
    if (sampledIndices.length >= maxPoints) break;
  }
  
  sampledIndices.sort((a, b) => a - b);
  
  const finalCount = Math.min(sampledIndices.length, maxPoints);
  console.log(`[Data] Sampled ${finalCount.toLocaleString()} points`);
  
  const newLats = new Float32Array(finalCount);
  const newLons = new Float32Array(finalCount);
  const newCanopy = new Uint8Array(finalCount);
  const newPopDensityLog = new Uint8Array(finalCount);
  const newLossYear = new Uint8Array(finalCount);
  const newPopCount = new Float32Array(finalCount);
  const newLossFrac = new Float32Array(finalCount);
  
  for (let i = 0; i < finalCount; i++) {
    const srcIdx = sampledIndices[i];
    newLats[i] = lats[srcIdx];
    newLons[i] = lons[srcIdx];
    newCanopy[i] = canopy[srcIdx];
    newPopDensityLog[i] = popDensityLog[srcIdx];
    newLossYear[i] = lossYear[srcIdx];
    newPopCount[i] = popCount[srcIdx];
    newLossFrac[i] = lossFrac[srcIdx];
  }
  
  return {
    totalPoints: finalCount,
    originalPoints: totalPoints,
    totalPopulation: totalPopulation,
    sampleRate: finalCount / totalPoints,
    lats: newLats,
    lons: newLons,
    canopy: newCanopy,
    popDensityLog: newPopDensityLog,
    lossYear: newLossYear,
    popCount: newPopCount,
    lossFrac: newLossFrac
  };
}

async function streamParseBinaryFile(filePath) {
  const stats = fs.statSync(filePath);
  const fileSize = stats.size;
  
  const headerBuffer = Buffer.alloc(4);
  const fd = fs.openSync(filePath, 'r');
  fs.readSync(fd, headerBuffer, 0, 4, 0);
  const totalPoints = headerBuffer.readUInt32LE(0);
  
  console.log(`[Data] File size: ${(fileSize / 1024 / 1024).toFixed(1)} MB`);
  console.log(`[Data] Total points in header: ${totalPoints.toLocaleString()}`);
  
  const lats = new Float32Array(totalPoints);
  const lons = new Float32Array(totalPoints);
  const canopy = new Uint8Array(totalPoints);
  const popDensityLog = new Uint8Array(totalPoints);
  const lossYear = new Uint8Array(totalPoints);
  const popCount = new Float32Array(totalPoints);
  const lossFrac = new Float32Array(totalPoints);
  
  const chunkPoints = Math.floor(READ_CHUNK_SIZE / BYTES_PER_POINT);
  let pointIndex = 0;
  let filePosition = 4;
  let totalPop = 0;
  
  console.log(`[Data] Reading in chunks of ${chunkPoints.toLocaleString()} points...`);
  
  while (pointIndex < totalPoints) {
    const pointsToRead = Math.min(chunkPoints, totalPoints - pointIndex);
    const bytesToRead = pointsToRead * BYTES_PER_POINT;
    const chunk = Buffer.alloc(bytesToRead);
    
    fs.readSync(fd, chunk, 0, bytesToRead, filePosition);
    
    let offset = 0;
    for (let i = 0; i < pointsToRead; i++) {
      const idx = pointIndex + i;
      lats[idx] = chunk.readFloatLE(offset); offset += 4;
      lons[idx] = chunk.readFloatLE(offset); offset += 4;
      canopy[idx] = chunk.readUInt8(offset); offset += 1;
      popDensityLog[idx] = chunk.readUInt8(offset); offset += 1;
      lossYear[idx] = chunk.readUInt8(offset); offset += 1;
      offset += 1;
      popCount[idx] = chunk.readFloatLE(offset); offset += 4;
      lossFrac[idx] = chunk.readFloatLE(offset); offset += 4;
      totalPop += popCount[idx];
    }
    
    pointIndex += pointsToRead;
    filePosition += bytesToRead;
    
    const progress = (pointIndex / totalPoints * 100).toFixed(0);
    if (pointIndex % Math.floor(totalPoints / 10) < chunkPoints) {
      console.log(`[Data] Progress: ${progress}% (${pointIndex.toLocaleString()} points)`);
    }
  }
  
  fs.closeSync(fd);
  
  return {
    totalPoints,
    totalPopulation: totalPop,
    lats,
    lons,
    canopy,
    popDensityLog,
    lossYear,
    popCount,
    lossFrac
  };
}

// ═══════════════════════════════════════════════════════════════
// VISUALIZATION DATA HANDLERS
// ═══════════════════════════════════════════════════════════════

ipcMain.handle('get-years', async () => {
  console.log(`[Data] Checking for years in: ${DATA_DIR}`);
  
  if (!fs.existsSync(DATA_DIR)) {
    console.log('[Data] Data directory not found');
    return [];
  }

  const years = [];
  for (const year of [2000, 2005, 2010, 2015, 2020]) {
    const binPath = path.join(DATA_DIR, `land_${year}.bin`);
    if (fs.existsSync(binPath)) {
      const stats = fs.statSync(binPath);
      const totalPoints = Math.floor((stats.size - 4) / BYTES_PER_POINT);
      years.push({ year, size: stats.size, totalPoints });
      console.log(`[Data] Found ${year}: ${totalPoints.toLocaleString()} points, ${(stats.size / 1024 / 1024).toFixed(1)} MB`);
    }
  }

  return years;
});

ipcMain.handle('load-year', async (event, year) => {
  const binPath = path.join(DATA_DIR, `land_${year}.bin`);
  
  console.log(`[Data] Loading year ${year} from: ${binPath}`);
  const startTime = Date.now();

  if (!fs.existsSync(binPath)) {
    throw new Error(`Data file not found: ${binPath}`);
  }

  try {
    const rawData = await streamParseBinaryFile(binPath);
    
    const parseTime = (Date.now() - startTime) / 1000;
    console.log(`[Data] Parsed ${rawData.totalPoints.toLocaleString()} points in ${parseTime.toFixed(2)}s`);
    console.log(`[Data] Total population: ${(rawData.totalPopulation / 1e9).toFixed(2)}B`);

    const data = downsampleForVisualization(rawData, MAX_VISUALIZATION_POINTS);
    
    const elapsed = (Date.now() - startTime) / 1000;
    console.log(`[Data] Total load time: ${elapsed.toFixed(2)}s`);

    const result = {
      totalPoints: data.totalPoints,
      originalPoints: data.originalPoints || data.totalPoints,
      totalPopulation: data.totalPopulation,
      loadTime: elapsed,
      sampleRate: data.sampleRate || 1.0,
      lats: Array.from(data.lats),
      lons: Array.from(data.lons),
      canopy: Array.from(data.canopy),
      popDensityLog: Array.from(data.popDensityLog),
      lossYear: Array.from(data.lossYear),
      popCount: Array.from(data.popCount),
      lossFrac: Array.from(data.lossFrac)
    };
    
    console.log(`[Data] Transferring ${result.totalPoints.toLocaleString()} points to renderer`);
    return result;
    
  } catch (e) {
    console.error(`[Data] Failed to load: ${e.message}`);
    console.error(e.stack);
    throw e;
  }
});

// ═══════════════════════════════════════════════════════════════
// TILE INDEX HANDLERS
// ═══════════════════════════════════════════════════════════════

ipcMain.handle('get-available-tiles', async () => {
  if (!fs.existsSync(TILES_DIR)) {
    console.log('[Tiles] Tiles directory not found:', TILES_DIR);
    return [];
  }

  const tiles = [];
  const tileDirs = fs.readdirSync(TILES_DIR);

  for (const tileName of tileDirs) {
    const tilePath = path.join(TILES_DIR, tileName);
    if (!fs.statSync(tilePath).isDirectory()) continue;

    const match = tileName.match(/(\d+)([NS])_(\d+)([EW])/);
    if (!match) continue;

    let lat = parseInt(match[1]);
    if (match[2] === 'S') lat = -lat;

    let lon = parseInt(match[3]);
    if (match[4] === 'W') lon = -lon;

    const years = [];
    for (const year of [2000, 2005, 2010, 2015, 2020]) {
      const csvPath = path.join(tilePath, `canopy_population_loss_${year}.csv`);
      if (fs.existsSync(csvPath)) {
        years.push(year);
      }
    }

    if (years.length > 0) {
      tiles.push({
        name: tileName,
        bounds: {
          minLat: lat - 10,
          maxLat: lat,
          minLon: lon,
          maxLon: lon + 10
        },
        years
      });
    }
  }

  console.log(`[Tiles] Indexed ${tiles.length} tiles`);
  return tiles;
});

ipcMain.handle('get-regions', async () => {
  return REGIONS;
});

// ═══════════════════════════════════════════════════════════════
// ANALYSIS HANDLERS - FIXED FOR M-CHIP MACS
// ═══════════════════════════════════════════════════════════════

function runPythonAnalysis(args) {
  return new Promise((resolve, reject) => {
    if (!PYTHON_PATH) {
      reject(new Error('Python is not available. Analysis features are disabled.'));
      return;
    }
    
    console.log(`[Analysis] Running: ${PYTHON_PATH} ${args.join(' ')}`);

    const pythonProcess = spawn(PYTHON_PATH, args, {
      cwd: __dirname,
      env: { 
        ...process.env, 
        PYTHONUNBUFFERED: '1',
        // Add Homebrew paths for M-chip Macs
        PATH: `/opt/homebrew/bin:/usr/local/bin:${process.env.PATH}`
      }
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
      const lines = data.toString().trim().split('\n');
      lines.forEach(line => {
        if (line.trim()) console.log(`[Python] ${line}`);
      });
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.error(`[Python Error] ${data.toString().trim()}`);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout, stderr, code });
      } else {
        reject(new Error(stderr || `Process exited with code ${code}`));
      }
    });

    pythonProcess.on('error', (err) => {
      reject(new Error(`Failed to start Python: ${err.message}`));
    });
  });
}

ipcMain.handle('run-point-analysis', async (event, params) => {
  const { lat, lon, radiusKm, year } = params;

  console.log(`[Analysis] Point analysis at (${lat.toFixed(4)}, ${lon.toFixed(4)}), radius ${radiusKm}km, year ${year}`);

  const outputDir = path.join(TEMP_DIR, `point_${Date.now()}`);
  fs.mkdirSync(outputDir, { recursive: true });

  try {
    const args = [
      ANALYSIS_SCRIPT,
      '--mode', 'point',
      '--lat', lat.toString(),
      '--lon', lon.toString(),
      '--radius', radiusKm.toString(),
      '--year', year.toString(),
      '--tiles-dir', TILES_DIR,
      '--output-dir', outputDir
    ];

    await runPythonAnalysis(args);

    const resultsPath = path.join(outputDir, 'results.json');
    if (fs.existsSync(resultsPath)) {
      const results = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
      console.log('[Analysis] Point analysis complete');
      return results;
    } else {
      throw new Error('Analysis completed but no results file found');
    }

  } catch (e) {
    console.error(`[Analysis] Failed: ${e.message}`);
    throw e;
  }
});

ipcMain.handle('run-region-analysis', async (event, params) => {
  const { regionId, year } = params;

  if (!REGIONS[regionId]) {
    throw new Error(`Unknown region: ${regionId}`);
  }

  const region = REGIONS[regionId];
  console.log(`[Analysis] Region analysis: ${region.name}, year ${year}`);

  const outputDir = path.join(TEMP_DIR, `region_${regionId}_${Date.now()}`);
  fs.mkdirSync(outputDir, { recursive: true });

  try {
    const args = [
      ANALYSIS_SCRIPT,
      '--mode', 'region',
      '--region', regionId,
      '--year', year.toString(),
      '--tiles-dir', TILES_DIR,
      '--regions-file', REGIONS_FILE,
      '--output-dir', outputDir
    ];

    await runPythonAnalysis(args);

    const resultsPath = path.join(outputDir, 'results.json');
    if (fs.existsSync(resultsPath)) {
      const results = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
      results.region = region;
      console.log('[Analysis] Region analysis complete');
      return results;
    } else {
      throw new Error('Analysis completed but no results file found');
    }

  } catch (e) {
    console.error(`[Analysis] Failed: ${e.message}`);
    throw e;
  }
});

ipcMain.handle('compare-regions', async (event, params) => {
  const { regionIds, year } = params;

  console.log(`[Analysis] Comparing ${regionIds.length} regions for year ${year}`);

  const results = {};

  for (const regionId of regionIds) {
    if (!REGIONS[regionId]) {
      results[regionId] = { error: `Unknown region: ${regionId}` };
      continue;
    }

    try {
      const outputDir = path.join(TEMP_DIR, `compare_${regionId}_${Date.now()}`);
      fs.mkdirSync(outputDir, { recursive: true });

      const args = [
        ANALYSIS_SCRIPT,
        '--mode', 'region',
        '--region', regionId,
        '--year', year.toString(),
        '--tiles-dir', TILES_DIR,
        '--regions-file', REGIONS_FILE,
        '--output-dir', outputDir
      ];

      await runPythonAnalysis(args);

      const resultsPath = path.join(outputDir, 'results.json');
      if (fs.existsSync(resultsPath)) {
        results[regionId] = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
        results[regionId].region = REGIONS[regionId];
      } else {
        results[regionId] = { error: 'No results file' };
      }

    } catch (e) {
      results[regionId] = { error: e.message };
    }
  }

  console.log('[Analysis] Comparison complete');
  return results;
});

ipcMain.handle('run-hotspot-scan', async (event, params) => {
  const { year, gridSize } = params;

  console.log(`[Analysis] Starting global hotspot scan for year ${year}`);

  const outputDir = path.join(TEMP_DIR, `hotspots_${Date.now()}`);
  fs.mkdirSync(outputDir, { recursive: true });

  try {
    const args = [
      ANALYSIS_SCRIPT,
      '--mode', 'hotspots',
      '--year', year.toString(),
      '--tiles-dir', TILES_DIR,
      '--output-dir', outputDir,
      '--grid-size', (gridSize || 5).toString()
    ];

    await runPythonAnalysis(args);

    const resultsPath = path.join(outputDir, 'hotspots.json');
    if (fs.existsSync(resultsPath)) {
      const results = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
      console.log(`[Analysis] Hotspot scan complete: found ${results.hotspots?.length || 0} hotspots`);
      return results;
    } else {
      throw new Error('Hotspot scan completed but no results file found');
    }

  } catch (e) {
    console.error(`[Analysis] Hotspot scan failed: ${e.message}`);
    throw e;
  }
});

// ═══════════════════════════════════════════════════════════════
// STARTUP LOGGING
// ═══════════════════════════════════════════════════════════════

console.log('═══════════════════════════════════════════════════════════════');
console.log('  Global Forest Analyzer - Electron Main Process');
console.log('  M-chip Mac Compatible Version');
console.log('═══════════════════════════════════════════════════════════════');
console.log(`  Platform: ${process.platform} (${process.arch})`);
console.log(`  Data directory: ${DATA_DIR}`);
console.log(`  Tiles directory: ${TILES_DIR}`);
console.log(`  Analysis script: ${ANALYSIS_SCRIPT}`);
console.log(`  Temp directory: ${TEMP_DIR}`);
console.log('═══════════════════════════════════════════════════════════════');
