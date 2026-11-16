FROM oven/bun:1 AS build
WORKDIR /app
COPY package.json bun.lock .npmrc ./
RUN bun install --frozen-lockfile
COPY . .
# Unterpfad hinter dem Reverse-Proxy, z.B. /hochschule/ma (wird beim Build eingebaut)
ARG BASE_PATH=
ENV BASE_PATH=$BASE_PATH
RUN bun run build

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3000 \
    BODY_SIZE_LIMIT=50M

COPY --from=build /app/build ./build
COPY --from=build /app/package.json ./
USER node
EXPOSE 3000
CMD ["node", "build"]
