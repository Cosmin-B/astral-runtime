import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const docsRoot = path.join(repositoryRoot, 'docs-site', 'docs');
const manifest = JSON.parse(await readFile(path.join(repositoryRoot, 'docs-site', 'source-manifest.json'), 'utf8'));

test('the Blume staging script publishes exactly the pinned 52 MkDocs pages', async () => {
  const result = spawnSync(process.execPath, ['scripts/prepare_blume_content.mjs'], {
    cwd: repositoryRoot,
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  assert.equal(manifest.files.length, 52);
  const pages = await collectMarkdown(docsRoot);
  assert.equal(pages.length, manifest.files.length);

  for (const entry of manifest.files) {
    const relativeTarget = entry.route ? `${entry.route}.md` : 'index.md';
    const target = path.join(docsRoot, relativeTarget);
    const output = await readFile(target, 'utf8');
    if (entry.route) assert.match(output, new RegExp(`^---\\ntitle: .+\\nslug: ${escapeRegExp(entry.route)}\\n---\\n`));
    else assert.match(output, /^---\ntitle: .+\n---\n/);
    assert.deepEqual(fencedPayloads(output), fencedPayloads(await readFile(path.join(repositoryRoot, entry.path), 'utf8')), entry.path);
  }

  const docsMap = await readFile(path.join(docsRoot, 'docs.md'), 'utf8');
  assert.match(docsMap, /\]\(\/docs\/api\/MEMORY_INDEX\)/);
  const overview = await readFile(path.join(docsRoot, 'index.md'), 'utf8');
  assert.match(overview, /https:\/\/github\.com\/Cosmin-B\/astral-runtime\/blob\//);
  assert.match(overview, /\[LICENSE\]\(https:\/\/github\.com\/Cosmin-B\/astral-runtime\/blob\/f2d13b77c70624ede5bc06823d4a794a4b955e10\/LICENSE\)/);
  assert.doesNotMatch(overview, /\]\(\/(?:LICENSE|NOTICE)\)/);
  const unity = await readFile(path.join(docsRoot, 'plugins', 'unity.md'), 'utf8');
  assert.match(unity, /\[LICENSE\]\(https:\/\/github\.com\/Cosmin-B\/astral-runtime\/blob\/f2d13b77c70624ede5bc06823d4a794a4b955e10\/LICENSE\)/);
  const config = await readFile(path.join(repositoryRoot, 'docs-site', 'blume.config.ts'), 'utf8');
  for (const entry of manifest.files) {
    const route = `/${entry.route}`;
    assert.ok(config.includes(`"${route}"`), `navigation is missing ${route}`);
  }
  assert.match(config, /base:\s*["']\/docs["']/);
  assert.equal(/basePath\s*:/.test(config), false);
  assert.match(config, /from: "\/docs\/integration\/UNITY_INTEGRATION", to: "\/plugins\/unity"/);
  assert.match(config, /from: "\/docs\/integration\/UNREAL_INTEGRATION", to: "\/plugins\/unreal\/AstralRT"/);
  assert.equal(JSON.stringify(manifest).includes('/Users/cosmin'), false);
});

test('overview and documentation map navigation use their logical content routes', async () => {
  const config = await readFile(path.join(repositoryRoot, 'docs-site', 'blume.config.ts'), 'utf8');
  assert.match(config, /const page = \(label: string, root: string\) => \(\{ label, root \}\)/);
  assert.match(config, /page\("Astral runtime",\s*"\/"\)/);
  assert.match(config, /page\("Documentation map",\s*"\/docs"\)/);

  const overview = await readFile(path.join(repositoryRoot, 'docs-site', 'dist', 'index.html'), 'utf8');
  const docsMap = await readFile(path.join(repositoryRoot, 'docs-site', 'dist', 'docs', 'index.html'), 'utf8');
  const overviewLink = linkForText(overview, 'Astral runtime');
  const docsMapLink = linkForText(docsMap, 'Documentation map');
  assert.deepEqual(overviewLink, { href: '/docs', current: 'page' });
  assert.deepEqual(docsMapLink, { href: '/docs/docs', current: 'page' });
});

test('legacy Astral heading fragments remain at their matching sections', async () => {
  const result = spawnSync(process.execPath, ['scripts/prepare_blume_content.mjs'], {
    cwd: repositoryRoot,
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);

  const overview = await readFile(path.join(docsRoot, 'index.md'), 'utf8');
  assert.match(overview, /^---\n[\s\S]+?\n---\n\n<a id="astral"><\/a>/);

  const batching = await readFile(stagedPathForSource('docs/api/CONTINUOUS_BATCHING.md'), 'utf8');
  assert.match(batching, /<a id="3-feed-decode"><\/a>\s*\n### 3\) Feed \+ decode/);

  const concurrency = await readFile(stagedPathForSource('docs/architecture/CONCURRENCY_MODEL.md'), 'utf8');
  const duplicateDesignAliases = [...concurrency.matchAll(/<a id="(design_\d+)"><\/a>\s*\n### Design/g)].map((match) => match[1]);
  assert.deepEqual(duplicateDesignAliases, ['design_1', 'design_2', 'design_3']);

  const anchorManifest = JSON.parse(await readFile(path.join(repositoryRoot, 'docs-site', 'legacy-heading-anchors.json'), 'utf8'));
  assert.equal(anchorManifest.pages.reduce((count, page) => count + page.anchors.length, 0), 68);
  for (const page of anchorManifest.pages) {
    const builtPage = path.join(repositoryRoot, 'docs-site', 'dist', page.route, 'index.html');
    const html = await readFile(builtPage, 'utf8');
    for (const anchor of page.anchors) {
      const exactId = new RegExp(`\\bid="${escapeRegExp(anchor.id)}"`, 'g');
      assert.equal([...html.matchAll(exactId)].length, 1, `${page.route}#${anchor.id} must be emitted exactly once`);
    }
  }
});

async function collectMarkdown(root) {
  const entries = await readdir(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const target = path.join(root, entry.name);
    if (entry.isDirectory()) files.push(...(await collectMarkdown(target)));
    if (entry.isFile() && entry.name.endsWith('.md')) files.push(target);
  }
  return files;
}

function fencedPayloads(markdown) {
  const payloads = [];
  let fence;
  let payload = [];
  for (const line of markdown.split(/\r?\n/)) {
    const opener = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (!fence && opener) {
      fence = opener[1];
      payload = [];
    } else if (fence && new RegExp(`^\\s{0,3}${fence[0]}{${fence.length},}\\s*$`).test(line)) {
      payloads.push(payload.join('\n'));
      fence = undefined;
    } else if (fence) {
      payload.push(line);
    }
  }
  return payloads;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function stagedPathForSource(sourcePath) {
  const entry = manifest.files.find((page) => page.path === sourcePath);
  assert.ok(entry, `missing manifest entry for ${sourcePath}`);
  return path.join(docsRoot, entry.route ? `${entry.route}.md` : 'index.md');
}

function linkForText(html, label) {
  for (const match of html.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/gi)) {
    const text = match[2].replace(/<[^>]*>/g, '').trim();
    if (text !== label) continue;
    const href = match[1].match(/\bhref="([^"]+)"/i)?.[1];
    const current = match[1].match(/\baria-current="([^"]+)"/i)?.[1];
    return { href, current };
  }
  return undefined;
}
