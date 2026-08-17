/**
 * Geographic lock for the Graph Leaflet map.
 *
 * Sources (not invented):
 * - Wikipedia "Pench Tiger Reserve": 21°41′35″N 79°14′54″E
 *   = 21.69306°N, 79.24833°E
 * - OpenStreetMap Nominatim (Pench National Park, osm way 155466998):
 *   center 21.6950552, 79.2487702
 *   boundingbox [south, north, west, east] =
 *   21.5384398, 21.8495193, 79.0845679, 79.3669939
 *
 * maxBounds pad the OSM national-park box by 0.12° (~13 km) so the
 * wider tiger-reserve buffer and entry gates stay inside the lock.
 * No reserve polygon is drawn: the project has no official boundary file.
 */

export const PENCH_CENTER: [number, number] = [21.69506, 79.24877]
export const PENCH_ZOOM = 11
export const PENCH_MIN_ZOOM = 10
export const PENCH_MAX_ZOOM = 16

/** Southwest [lat, lon], northeast [lat, lon]. */
export const PENCH_BOUNDS: [[number, number], [number, number]] = [
  [21.41844, 78.96457],
  [21.96952, 79.48699],
]

export const PENCH_MAP_LABEL = 'Pench Tiger Reserve — Wildlife Intelligence'
