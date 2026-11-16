import { proxy } from '$lib/server/proxy';
import type { RequestHandler } from './$types';

/** /api/<pfad> -> Python-Service */
const handler: RequestHandler = ({ params, request, url }) => proxy(params.path, request, url);

export const GET = handler;
export const POST = handler;
