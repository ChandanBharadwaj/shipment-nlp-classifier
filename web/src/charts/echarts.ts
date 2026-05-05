// echarts.ts — single registration point for the ECharts modules we use.
//
// vue-echarts requires us to opt-in to chart types and components so the
// final bundle only ships what the app actually renders. Importing this
// file once (from main.ts or VChart.vue) registers everything globally;
// after that any view can compose <VChart :option="…"/> without further
// imports. Adding a new chart type? Register it here, not in the view.
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { BarChart, ScatterChart, HeatmapChart, LineChart } from "echarts/charts";
import {
  TooltipComponent,
  GridComponent,
  LegendComponent,
  VisualMapComponent,
  DataZoomComponent,
  TitleComponent,
  MarkLineComponent,
} from "echarts/components";

use([
  CanvasRenderer,
  BarChart,
  ScatterChart,
  HeatmapChart,
  LineChart,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  VisualMapComponent,
  DataZoomComponent,
  TitleComponent,
  MarkLineComponent,
]);
