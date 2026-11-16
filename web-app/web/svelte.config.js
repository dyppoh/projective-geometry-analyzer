import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { loadEnv } from 'vite';

// Unterpfad, unter dem die App läuft (z.B. /hochschule/ma) - wird beim Build fest eingebaut.
// Aus der Umgebung (Docker: Build-Argument) oder der .env, sonst Wurzel.
const { BASE_PATH = '' } = loadEnv(process.env.NODE_ENV ?? '', '.', '');
const base = BASE_PATH.trim().replace(/\/+$/, '').replace(/^(?=[^/])/, '/');

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	compilerOptions: { runes: true },
	kit: { adapter: adapter(), paths: { base } }
};

export default config;
