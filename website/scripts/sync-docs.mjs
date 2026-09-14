/**
 * Publish the public documentation to the docs site.
 *
 * `docs/` at the repository root stays the single source of truth; this copies
 * the public subset into Starlight's content collection, so the site never
 * holds a second copy that can drift.
 *
 * Two things it does beyond copying:
 *
 *   - `docs/README.md` becomes `docs/index.md`, which is what Starlight serves
 *     at `/docs/`.
 *   - A link to a page that is NOT published is unwrapped to its text. The
 *     index points at the internal reports and the frozen paper sources, and a
 *     dead link on a public page is worse than plain words.
 *
 * Run automatically by `bun run dev` / `bun run build`; see package.json.
 */

import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { base } from '../site.config.mjs';

const website = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(website, '..');
const source = path.join(repo, 'docs');
const target = path.join(website, 'src', 'content', 'docs', 'docs');

/** Published pages, relative to `docs/`. Anything else stays off the site. */
const PUBLISHED = [
  'architecture.md',
  'reproducibility.md',
  'README.md',
  'guides/commands.md',
  'guides/running-models.md',
  'guides/rust-and-devices.md',
  'guides/training.md',
  'reference/cli.md',
  'reference/datasets.md',
  'reference/models.md',
  'reference/modules.md',
  'reference/pipelines.md',
  'audit/AUDIT_CHECKLIST.md',
  'audit/MODEL_CARDS.md',
];

const published = new Set(PUBLISHED);
const isPublished = (rel) => published.has(rel);

/** `docs/README.md` is the index of `/docs/`; everything else keeps its path. */
const destinationOf = (rel) => (rel === 'README.md' ? 'index.md' : rel);

/**
 * Starlight takes the page title from frontmatter and requires it, so the
 * document's own `# Heading` is moved there and removed from the body — leaving
 * it in place would render the title twice.
 */
function withTitle(markdown, rel) {
  const heading = markdown.match(/^#[ \t]+(.+?)[ \t]*$/m);
  if (!heading) {
    throw new Error(`sync-docs: ${rel} has no "# Heading", so it has no page title`);
  }
  const title = JSON.stringify(heading[1].trim());
  const body = markdown.replace(heading[0], '').replace(/^\s*\n/, '');
  return `---\ntitle: ${title}\n---\n\n${body}`;
}

// [text](target) — every markdown link; the target decides what happens to it.
const LINK = /\[([^\]]+)\]\(([^)\s]+)\)/g;

function rewriteLinks(markdown, rel) {
  const fromDir = path.posix.dirname(rel);
  return markdown.replace(LINK, (whole, text, href) => {
    // Absolute URLs, site-absolute paths and in-page anchors are none of our
    // business.
    if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith('/') || href.startsWith('#')) {
      return whole;
    }
    const [target, anchor = ''] = href.split(/(?=#)/);
    const resolved = path.posix
      .normalize(path.posix.join(fromDir, target))
      .replace(/^\.\//, '');
    // A link to a published page keeps working, pointed at its published path
    // (README becomes index). Anything else — an internal report, a frozen
    // paper directory — is unwrapped to its text rather than left dangling.
    if (resolved.endsWith('.md') && isPublished(resolved)) {
      const to = path.posix.relative(fromDir, destinationOf(resolved));
      return `[${text}](${to}${anchor})`;
    }
    return text;
  });
}

await rm(target, { recursive: true, force: true });

/** The URL Starlight serves a published page at, e.g. `docs/audit/audit_checklist/`. */
function routeOf(rel) {
  const dir = path.posix.dirname(rel);
  const slug = path.posix.basename(rel, '.md').toLowerCase();
  return dir === '.' ? `docs/${slug}/` : `docs/${dir}/${slug}/`;
}

let written = 0;
for (const rel of PUBLISHED) {
  const body = await readFile(path.join(source, rel), 'utf8');
  const dest = path.join(target, destinationOf(rel));
  await mkdir(path.dirname(dest), { recursive: true });
  await writeFile(dest, withTitle(rewriteLinks(body, rel), rel));
  written += 1;
}

// Anyone who knows the repository layout will try the file path, so route it to
// the page. Only paths that differ from the page's own URL become redirects —
// `/docs/architecture` IS the page, and a redirect there is a route conflict
// that stops the page rendering.
const redirects = {};
for (const rel of PUBLISHED) {
  if (rel === 'README.md') continue;
  // Astro matches redirect keys against the path with `base` already stripped,
  // so the collision check is base-relative; the target needs the base, because
  // Astro does not apply it there.
  const ownRoute = `/${routeOf(rel)}`;
  const route = `${base}${ownRoute}`;
  const ownRouteNoSlash = ownRoute.replace(/\/$/, '');
  const stem = rel.replace(/\.md$/, '');
  const candidates = [
    `docs/${rel}`, // docs/audit/AUDIT_CHECKLIST.md
    `docs/${rel.toLowerCase()}`, // docs/audit/audit_checklist.md
    `docs/${stem}`, // docs/audit/AUDIT_CHECKLIST
    `docs/${stem.toLowerCase()}`, // docs/audit/audit_checklist
  ];
  for (const name of candidates) {
    const key = `/${name}`;
    if (key === ownRouteNoSlash || key === ownRoute) continue;
    redirects[key] = route;
  }
}
await writeFile(
  path.join(website, 'src', 'redirects.mjs'),
  `// Generated by scripts/sync-docs.mjs — edit PUBLISHED there, not this file.\n` +
    `export default ${JSON.stringify(redirects, null, 2)};\n`,
);

console.log(
  `sync-docs: published ${written} page(s) and ${Object.keys(redirects).length} redirect(s) from docs/`,
);
