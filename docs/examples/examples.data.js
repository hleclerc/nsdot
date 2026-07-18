import { createContentLoader } from 'vitepress'
import tagWeights from './tag-weights.json'

export default createContentLoader( 'examples/*.md', {
  transform( rawData ) {
    const examples = rawData
      .filter( p => !p.url.match( /\/examples\/?$/ ) )
      .map( p => {
        const slug = p.url.replace( /\/$/, '' ).split( '/' ).pop()
        return {
          title:        p.frontmatter.title        ?? slug,
          description:  p.frontmatter.description  ?? '',
          tags:         p.frontmatter.tags         ?? [],
          order:        p.frontmatter.order        ?? 999,
          tutorialLink: p.frontmatter.tutorialLink ?? null,
          link:         p.url,
          image:        slug + '.png',
        }
      } )
      .sort( ( a, b ) => a.order - b.order || a.title.localeCompare( b.title ) )

    // Count how many examples carry each tag
    const freq = {}
    examples.forEach( e => e.tags.forEach( t => { freq[ t ] = ( freq[ t ] ?? 0 ) + 1 } ) )

    // Collect unique tags and sort:
    //   1. explicit weight from tag-weights.json (ascending)
    //   2. unweighted tags (weight = Infinity) sorted by frequency desc, then alpha
    const tags = [ ...new Set( examples.flatMap( e => e.tags ) ) ]
      .sort( ( a, b ) => {
        const wa = tagWeights[ a ] ?? Infinity
        const wb = tagWeights[ b ] ?? Infinity
        if ( wa !== wb ) return wa - wb
        const diff = ( freq[ b ] ?? 0 ) - ( freq[ a ] ?? 0 )
        return diff !== 0 ? diff : a.localeCompare( b )
      } )

    return { examples, tags }
  }
} )
