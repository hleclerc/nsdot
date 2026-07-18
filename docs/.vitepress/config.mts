import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  title: "SDOT",
  description: "Fast, differentiable, N-dimensional Semi-Discrete Optimal Transport — with JAX and PyTorch.",
  base: '/sdot/',
  markdown: {
    config: ( md ) => { md.use( mathjax3 ) }
  },
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: ( tag ) => tag.startsWith( 'mjx-' )
      }
    }
  },
  themeConfig: {
    nav: [
      { text: 'Guide',      link: '/guide/getting-started' },
      { text: 'Tutorials',  link: '/tutorials/ot-plan-intro' },
      { text: 'Examples',   link: '/examples/' },
      { text: 'API',        link: '/api/distributions' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Getting Started',          link: '/guide/getting-started' },
            { text: 'Distributions',            link: '/guide/distributions' },
            { text: 'Ground Metrics',           link: '/guide/ground-metrics' },
            { text: 'Backends (JAX / PyTorch)', link: '/guide/backends' },
            { text: 'Differentiability',        link: '/guide/differentiability' },
            { text: 'Ot Plans',                 link: '/guide/ot-plan' },
          ]
        }
      ],
      '/tutorials/': [
        {
          text: 'Tutorials',
          items: [
            { text: 'Optimal Transport Plans', link: '/tutorials/ot-plan-intro' },
            { text: 'Gaussian Mixture Fit',    link: '/tutorials/gaussian-mixture-fit' },
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Examples',
          items: [
            { text: 'Gallery',               link: '/examples/' },
            { text: 'Uniform Fill',          link: '/examples/uniform-fill' },
            { text: 'Image Transport',       link: '/examples/image-transport' },
            { text: 'Lloyd Quantization',    link: '/examples/lloyd-quantization' },
            { text: 'Gaussian Mixture Fit',  link: '/examples/gaussian-mixture-fit' },
            { text: 'Power Diagram Geometry',link: '/examples/power-diagram-geometry' },
            { text: 'Periodic Metrics',      link: '/examples/periodic-metrics' },
            { text: 'Crowd Motion',          link: '/examples/crowd-motion' },
            { text: 'Incompressible Euler',  link: '/examples/euler-equations' },
            { text: 'Fokker-Planck',         link: '/examples/fokker-planck' },
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Distributions',  link: '/api/distributions' },
            { text: 'OT Plan',        link: '/api/ot-plan' },
            { text: 'PowerDiagram',   link: '/api/power-diagram' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/sdot-team/sdot' }
    ],

    search: { provider: 'local' },

    editLink: {
      pattern: 'https://github.com/sdot-team/sdot/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    footer: {
      message: 'SDOT — H. Leclerc, Q. Mérigot, T. Gallouët · LMO / INRIA PARMA',
    }
  }
})
