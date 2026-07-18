import DefaultTheme from 'vitepress/theme'
import ExamplesGallery from './ExamplesGallery.vue'

export default {
  extends: DefaultTheme,
  enhanceApp( { app } ) {
    app.component( 'ExamplesGallery', ExamplesGallery )
  }
}
