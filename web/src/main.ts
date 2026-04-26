// main.ts — Vue app entry point.
//
// Order matters: tokens.css → base.css must load before any component
// renders so CSS variables are defined when scoped styles reference them.
import { createApp } from "vue";
import { createPinia } from "pinia";

import App from "./App.vue";
import { router } from "./router";

import "./styles/tokens.css";
import "./styles/base.css";

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");
