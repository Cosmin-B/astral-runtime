import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const docsSiteRoot = path.join(repositoryRoot, 'docs-site');
const docsRoot = path.join(docsSiteRoot, 'docs');
const manifestPath = path.join(docsSiteRoot, 'source-manifest.json');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const pagesBySource = new Map(manifest.files.map((entry) => [entry.path, entry]));
const legacyAnchorManifest = JSON.parse(await readFile(path.join(docsSiteRoot, 'legacy-heading-anchors.json'), 'utf8'));
const anchorsByRoute = new Map(legacyAnchorManifest.pages.map((page) => [page.route, page.anchors]));
const manifestRoutes = new Set(manifest.files.map((entry) => entry.route ?? ''));

for (const [route, anchors] of anchorsByRoute) {
  if (!manifestRoutes.has(route)) throw new Error(`Legacy Astral anchors reference an unpublished route: ${route}`);
  const ids = new Set();
  for (const anchor of anchors) {
    if (!/^[a-z0-9_-]+$/.test(anchor.id) || !/^[a-z0-9]+$/.test(anchor.headingKey) || !Number.isInteger(anchor.occurrence) || anchor.occurrence < 1) {
      throw new Error(`Invalid legacy Astral anchor metadata for route ${route}: ${JSON.stringify(anchor)}`);
    }
    if (ids.has(anchor.id)) throw new Error(`Duplicate legacy Astral anchor ${anchor.id} on route ${route}`);
    ids.add(anchor.id);
  }
}

if (new Set(manifest.files.map((entry) => entry.route)).size !== manifest.files.length) {
  throw new Error('The Astral source manifest contains duplicate routes.');
}

await rm(docsRoot, { recursive: true, force: true });
await mkdir(docsRoot, { recursive: true });
await mkdir(path.join(docsSiteRoot, 'public'), { recursive: true });

for (const entry of manifest.files) {
  const sourcePath = path.join(repositoryRoot, entry.path);
  const source = await readFile(sourcePath, 'utf8');
  const digest = createHash('sha256').update(source).digest('hex');
  if (digest !== entry.sha256) throw new Error(`Source changed from pinned ${manifest.revision}: ${entry.path}`);

  const title = getTitle(source, entry.path);
  const body = transformMarkdown(source, entry, anchorsByRoute.get(entry.route ?? '') ?? []);
  const metadata = `---\ntitle: ${JSON.stringify(title)}\n${entry.route ? `slug: ${entry.route}\n` : ''}---\n\n`;
  const sourceUrl = `https://github.com/Cosmin-B/astral-runtime/blob/${manifest.revision}/${githubPath(entry.path)}`;
  const editUrl = `https://github.com/Cosmin-B/astral-runtime/edit/main/${githubPath(entry.path)}`;
  const output = `${metadata}${body.trimEnd()}\n\nSource: [View the pinned source](${sourceUrl}) · [Edit this source](${editUrl})\n`;
  const destination = path.join(docsRoot, entry.route ? `${entry.route}.md` : 'index.md');
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, output, 'utf8');
}

for (const filename of ['LICENSE', 'NOTICE']) {
  await copyFile(path.join(repositoryRoot, filename), path.join(docsSiteRoot, 'public', filename));
}

console.log(`Staged ${manifest.files.length} allowlisted Astral pages for Blume ${manifest.revision}.`);

function getTitle(markdown, sourcePath) {
  const header = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  const frontmatterTitle = header?.[1].match(/^title:\s*(.*?)\s*$/m)?.[1];
  if (frontmatterTitle) return frontmatterTitle.replace(/^['"]|['"]$/g, '');
  const lines = markdown.split(/\r?\n/);
  let fenced = false;
  for (const line of lines) {
    if (/^\s{0,3}(`{3,}|~{3,})/.test(line)) fenced = !fenced;
    if (!fenced) {
      const heading = line.match(/^#\s+(.+?)\s*#*\s*$/);
      if (heading) return heading[1].replace(/`/g, '').replace(/\[([^\]]+)\]\([^)]*\)/g, '$1').trim();
    }
  }
  return path.posix.basename(sourcePath, path.posix.extname(sourcePath)).replaceAll('_', ' ');
}

function transformMarkdown(markdown, currentEntry, legacyAnchors) {
  const output = [];
  let fence;
  let removedTitle = false;
  const headingOccurrences = new Map();
  for (const line of markdown.split(/\r?\n/)) {
    const opener = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (!fence && opener) {
      fence = opener[1];
      output.push(line);
      continue;
    }
    if (fence) {
      output.push(line);
      const closing = new RegExp(`^\\s{0,3}${fence[0]}{${fence.length},}\\s*$`);
      if (closing.test(line)) fence = undefined;
      continue;
    }

    const heading = line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const headingKey = normalizeHeadingKey(heading[1]);
      const occurrence = (headingOccurrences.get(headingKey) ?? 0) + 1;
      headingOccurrences.set(headingKey, occurrence);
      for (const anchor of legacyAnchors) {
        if (anchor.headingKey === headingKey && anchor.occurrence === occurrence) {
          output.push(`<a id="${anchor.id}"></a>`);
        }
      }
    }

    if (!removedTitle && /^#\s+/.test(line) && line.replace(/^#\s+/, '').trim() === getTitle(markdown, currentEntry.path)) {
      removedTitle = true;
      continue;
    }
    output.push(rewriteReferences(line, currentEntry));
  }
  return output.join('\n').replace(/\n{3,}/g, '\n\n');
}

function normalizeHeadingKey(value) {
  return value
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/`/g, '')
    .replace(/<[^>]*>/g, '')
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');
}

function rewriteReferences(line, currentEntry) {
  return line
    .replace(/(!?\[[^\]]*\]\()(<[^>]+>|[^)\s]+)([^)]*\))/g, (whole, prefix, targetToken, suffix) => {
      const wrapped = targetToken.startsWith('<') && targetToken.endsWith('>');
      const target = wrapped ? targetToken.slice(1, -1) : targetToken;
      const rewritten = rewriteTarget(target, currentEntry);
      return `${prefix}${wrapped ? `<${rewritten}>` : rewritten}${suffix}`;
    })
    .replace(/(\b(?:href|src)=['"])([^'"]+)(['"])/gi, (whole, prefix, target, quote) => `${prefix}${rewriteTarget(target, currentEntry)}${quote}`)
    .replace(/(^\s*\[[^\]]+\]:\s+)(<[^>]+>|\S+)(.*$)/, (whole, prefix, targetToken, suffix) => {
      const wrapped = targetToken.startsWith('<') && targetToken.endsWith('>');
      const target = wrapped ? targetToken.slice(1, -1) : targetToken;
      const rewritten = rewriteTarget(target, currentEntry);
      return `${prefix}${wrapped ? `<${rewritten}>` : rewritten}${suffix}`;
    });
}

function rewriteTarget(target, currentEntry) {
  if (!target || target.startsWith('#') || /^(?:https?:|mailto:|tel:|data:|javascript:|\/\/)/i.test(target)) return target;
  const splitAt = target.search(/[?#]/);
  const targetPath = splitAt === -1 ? target : target.slice(0, splitAt);
  const suffix = splitAt === -1 ? '' : target.slice(splitAt);
  if (!targetPath) return target;
  const rootLegalFile = targetPath.match(/^\/(LICENSE|NOTICE)$/);
  if (rootLegalFile) return `${githubUrl(rootLegalFile[1])}${suffix}`;

  const sourceTarget = path.posix.normalize(path.posix.join(path.posix.dirname(currentEntry.path), decodeURIComponent(targetPath)));
  const markdownTarget = sourceTarget.endsWith('.md') || sourceTarget.endsWith('.mdx')
    ? sourceTarget
    : path.posix.extname(sourceTarget) ? '' : `${sourceTarget}.md`;
  const doc = pagesBySource.get(markdownTarget);
  if (doc) return `${routeUrl(doc.route)}${suffix}`;

  if (['docs/integration/UNITY_INTEGRATION.md', 'docs/integration/UNREAL_INTEGRATION.md'].includes(sourceTarget)) {
    return `${sourceTarget.endsWith('UNITY_INTEGRATION.md') ? '/plugins/unity' : '/plugins/unreal/AstralRT'}${suffix}`;
  }
  if (sourceTarget === 'LICENSE' || sourceTarget === 'NOTICE') return `${githubUrl(sourceTarget)}${suffix}`;
  if (sourceTarget.startsWith('../') || sourceTarget.startsWith('/')) return target;

  const absoluteSource = path.join(repositoryRoot, sourceTarget);
  if (!existsSync(absoluteSource)) return target;
  return `${githubUrl(sourceTarget)}${suffix}`;
}

function routeUrl(route) {
  return route ? `/${route}` : '/';
}

function githubUrl(sourcePath) {
  return `https://github.com/Cosmin-B/astral-runtime/blob/${manifest.revision}/${githubPath(sourcePath)}`;
}

function githubPath(value) {
  return value.split('/').map((part) => encodeURIComponent(part)).join('/');
}
