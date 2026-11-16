import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
	// Dev-Server: Anfragen über die öffentliche Adresse (Tunnel) zulassen
	const { PUBLIC_URL } = loadEnv(mode, '.', '');
	return {
		plugins: [sveltekit()],
		server: { allowedHosts: PUBLIC_URL ? [new URL(PUBLIC_URL).hostname] : [] }
	};
});
