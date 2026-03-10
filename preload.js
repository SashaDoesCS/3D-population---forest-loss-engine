/**
 * Electron Preload Script - Forest Analyzer
 * ==========================================
 * 
 * Exposes secure IPC channels to the renderer process for:
 * - Visualization data loading
 * - Analysis execution
 * - Region management
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('forestData', {
  // ═══════════════════════════════════════════════════════════════
  // VISUALIZATION DATA
  // ═══════════════════════════════════════════════════════════════
  
  /**
   * Get list of available years with visualization data
   * @returns {Promise<Array<{year: number, size: number, totalPoints: number}>>}
   */
  getYears: () => ipcRenderer.invoke('get-years'),
  
  /**
   * Load visualization data for a specific year
   * @param {number} year - Year to load
   * @returns {Promise<Object>} Visualization data arrays
   */
  loadYear: (year) => ipcRenderer.invoke('load-year', year),
  
  // ═══════════════════════════════════════════════════════════════
  // TILE & REGION INFO
  // ═══════════════════════════════════════════════════════════════
  
  /**
   * Get list of available tiles with their bounds
   * @returns {Promise<Array<{name: string, bounds: Object, years: number[]}>>}
   */
  getAvailableTiles: () => ipcRenderer.invoke('get-available-tiles'),
  
  /**
   * Get predefined analysis regions
   * @returns {Promise<Object>} Map of region ID to region definition
   */
  getRegions: () => ipcRenderer.invoke('get-regions'),
  
  // ═══════════════════════════════════════════════════════════════
  // ANALYSIS
  // ═══════════════════════════════════════════════════════════════
  
  /**
   * Run analysis on a specific point with radius
   * @param {Object} params
   * @param {number} params.lat - Latitude
   * @param {number} params.lon - Longitude
   * @param {number} params.radiusKm - Analysis radius in km
   * @param {number} params.year - Year to analyze
   * @returns {Promise<Object>} Analysis results
   */
  runPointAnalysis: (params) => ipcRenderer.invoke('run-point-analysis', params),
  
  /**
   * Run analysis on a predefined region
   * @param {Object} params
   * @param {string} params.regionId - Region identifier
   * @param {number} params.year - Year to analyze
   * @returns {Promise<Object>} Analysis results
   */
  runRegionAnalysis: (params) => ipcRenderer.invoke('run-region-analysis', params),
  
  /**
   * Compare multiple regions
   * @param {Object} params
   * @param {string[]} params.regionIds - Array of region identifiers
   * @param {number} params.year - Year to analyze
   * @returns {Promise<Object>} Comparison results keyed by region ID
   */
  compareRegions: (params) => ipcRenderer.invoke('compare-regions', params),

  /**
   * Run global hotspot scan to identify areas of research interest
   * @param {Object} params
   * @param {number} params.year - Year to analyze
   * @param {number} params.gridSize - Grid size in degrees (default 5)
   * @returns {Promise<Object>} Hotspot results
   */
  runHotspotScan: (params) => ipcRenderer.invoke('run-hotspot-scan', params),
});

console.log('[Preload] Forest Analyzer API exposed');
