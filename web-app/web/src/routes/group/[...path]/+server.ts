import { proxy } from '$lib/server/proxy';
import type { RequestHandler } from './$types';

/** /group/<token>/... -> Python-Service: Aufgabenseite, Bilder und Abgabe für das MLLM (öffentlich) */
const handler: RequestHandler = ({ params, request, url }) => proxy(`group/${params.path}`, request, url);

export const GET = handler;
export const POST = handler;
