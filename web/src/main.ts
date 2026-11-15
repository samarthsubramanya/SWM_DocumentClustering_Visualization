import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'

// Material Web components, imported one by one to keep the bundle small.
import '@material/web/button/filled-button.js'
import '@material/web/button/filled-tonal-button.js'
import '@material/web/button/outlined-button.js'
import '@material/web/button/text-button.js'
import '@material/web/chips/assist-chip.js'
import '@material/web/chips/chip-set.js'
import '@material/web/chips/suggestion-chip.js'
import '@material/web/divider/divider.js'
import '@material/web/focus/md-focus-ring.js'
import '@material/web/icon/icon.js'
import '@material/web/iconbutton/filled-icon-button.js'
import '@material/web/iconbutton/icon-button.js'
import '@material/web/progress/circular-progress.js'
import '@material/web/progress/linear-progress.js'
import '@material/web/ripple/ripple.js'
import '@material/web/select/outlined-select.js'
import '@material/web/select/select-option.js'
import '@material/web/switch/switch.js'
import '@material/web/textfield/outlined-text-field.js'

import App from './App.vue'
import { startTheme } from './lib/theme'
import './styles.css'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/explore' },
    { path: '/explore', component: () => import('./views/ExploreView.vue'), meta: { title: 'Explore' } },
    { path: '/ask', component: () => import('./views/AskView.vue'), meta: { title: 'Ask the agents' } },
  ],
})

startTheme()
createApp(App).use(router).mount('#app')
