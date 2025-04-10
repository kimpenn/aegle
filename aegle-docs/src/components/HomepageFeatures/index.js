import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Segmentation',
    Svg: require('@site/static/img/segmentation_chop.svg').default,
    description: (
      <>
        Based on DellCell Mesmer.
      </>
    ),
  },
  {
    title: 'Cell Profiling',
    Svg: require('@site/static/img/marker_violin.svg').default,
    description: (
      <>
        Extract mean antibody intensities.
      </>
    ),
  },
  {
    title: 'Downstream Analysis',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Clustering;
        LLM agent for cell type annotation;
        Pixel-level, Cell-level, and Patch-level statitcs
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
