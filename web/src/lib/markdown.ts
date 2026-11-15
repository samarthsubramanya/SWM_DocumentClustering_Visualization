// Markdown from agent answers and evaluation reports. Agent answers are LLM output, so the
// HTML is sanitized before it reaches the page.
import DOMPurify from 'dompurify'
import { marked } from 'marked'

export function renderMarkdown(text: string): string {
  return DOMPurify.sanitize(marked.parse(text, { async: false, gfm: true }) as string)
}
