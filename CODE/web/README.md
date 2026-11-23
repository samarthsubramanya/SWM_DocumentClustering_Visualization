# Web dashboard (Vue 3 + Material 3)

A second front end for the project, next to the Streamlit app in `ui/` (which stays as it is).
It is built with Vue 3, Vite and Google's Material Web components. The interface uses
Material's monochrome (black-and-white) scheme, in light and dark, generated with
`@material/material-color-utilities`; color is used only for data (clusters and entity types).

| View | What it shows |
|---|---|
| **Explore** | The run's scores and the map of posts. Click a post to see the selected cluster's terms, label and knowledge graph, and why the post is in that cluster: its evidence, the path to the cluster's core entities, and the rationale. |
| **Ask** | A chat with the harness agents. Knowledge-graph answers include the generated Cypher, any repairs and the result rows. |

## Run it

The web app talks to the FastAPI app, which serves the chat (`/api/v1/query`) and read-only
dashboard data (`/api/v1/explore/*`). Start the API from the `CODE/` folder:

```bash
uvicorn src.api.app:app --port 8000
```

Then, in `web/`:

```bash
npm install
npm run dev        # http://localhost:5173; /api is proxied to port 8000
```

`npm run build` type-checks and writes a static build to `web/dist/`.

## Notes

- **Post ids:** posts are shown by opaque ids (`doc-3f9a1c2e`). Real ids and newsgroup names
  give the gold label away, so they appear only with the **Gold labels** switch on.
- **Map controls:** drag to pan, Ctrl/⌘ + scroll or pinch to zoom, click a dot to open a post.
  Plain scrolling scrolls the page.
- **Navigation:** a rail on medium and wider windows, a bottom bar on phones.
