import { error } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';
import { env as publicEnv } from '$env/dynamic/public';
import { base as basePath } from '$app/paths';

/** Reicht eine Anfrage unverändert an den Python-Service durch (Status, Content-Type, Body - auch Bilder). */
export async function proxy(path: string, request: Request, url: URL): Promise<Response> {
	const base = (env.PYTHON_API_URL || 'http://127.0.0.1:8765').replace(/\/$/, '');
	let res: Response;
	try {
		res = await fetch(`${base}/${path}${url.search}`, {
			method: request.method,
			headers: {
				'Content-Type': request.headers.get('Content-Type') ?? 'application/json',
				// öffentliche Adresse für die Links auf der Aufgabenseite des MLLM
				'X-Public-Origin': publicEnv.PUBLIC_URL || url.origin + basePath
			},
			body: request.method === 'GET' ? undefined : await request.arrayBuffer()
		});
	} catch {
		error(503, `Python-Service unter ${base} nicht erreichbar.`);
	}
	return new Response(res.body, {
		status: res.status,
		headers: { 'Content-Type': res.headers.get('Content-Type') ?? 'application/json', 'Cache-Control': 'no-store' }
	});
}
