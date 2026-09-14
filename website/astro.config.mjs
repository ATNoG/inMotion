import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://atnog.github.io',
  base: '/inMotion',
  output: 'static',
  integrations: [
    starlight({
      title: 'inMotion',
      description:
        'Documentation for the inMotion edge inference platform: how it works, how to run it, and how to verify it.',
      // The published pages are synced from ../docs by scripts/sync-docs.mjs.
      // They sit under `docs/` in the collection so they are served at
      // /inMotion/docs/... and do not collide with the project pages, which
      // keep their own layout.
      sidebar: [
        {
          label: 'Start here',
          items: [
            { label: 'Documentation index', link: '/docs/' },
            { label: 'Architecture', link: '/docs/architecture/' },
            { label: 'Reproducibility', link: '/docs/reproducibility/' },
          ],
        },
        { label: 'Guides', items: [{ autogenerate: { directory: 'docs/guides' } }] },
        { label: 'Reference', items: [{ autogenerate: { directory: 'docs/reference' } }] },
        { label: 'Audit', items: [{ autogenerate: { directory: 'docs/audit' } }] },
      ],
    }),
  ],
});
