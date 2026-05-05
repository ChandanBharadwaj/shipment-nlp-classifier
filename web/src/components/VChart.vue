<script setup lang="ts">
/**
 * VChart.vue — thin wrapper around vue-echarts with the module registry
 * imported as a side-effect. Views just pass an `option` and a height.
 *
 * Why a wrapper:
 *   1. Centralise the echarts module registration (charts/echarts.ts) so
 *      tree-shaking stays effective — views never import echarts directly.
 *   2. Apply a consistent default height + autoresize so charts don't
 *      collapse when their parent flexes.
 *
 * vue-echarts surfaces all ECharts events as native Vue events on the
 * component (`@click`, `@mouseover`, etc.); we forward `@click` through
 * an explicit emit so view code gets type-safe params. Other events can
 * be wired the same way as needed.
 */
import "@/charts/echarts";
import VueECharts from "vue-echarts";

defineProps<{
  option: Record<string, unknown>;
  loading?: boolean;
  height?: string;
}>();

const emit = defineEmits<{
  (
    e: "click",
    params: {
      name?: string;
      value?: unknown;
      /** The full datum object passed in `series.data[i]` — includes any
       *  custom fields (e.g. `meta`) the caller attached. */
      data?: unknown;
      dataIndex?: number;
      seriesName?: string;
    },
  ): void;
}>();

function onClick(params: unknown): void {
  emit("click", params as {
    name?: string;
    value?: unknown;
    data?: unknown;
    dataIndex?: number;
    seriesName?: string;
  });
}
</script>

<template>
  <div class="v-chart-wrap" :style="{ height: height || '320px' }">
    <VueECharts
      class="v-chart"
      :option="option"
      :loading="loading"
      autoresize
      @click="onClick"
    />
  </div>
</template>

<style scoped>
.v-chart-wrap {
  width: 100%;
  position: relative;
}
.v-chart {
  width: 100%;
  height: 100%;
}
</style>
