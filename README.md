# POC de Detección de Fraude en Transacciones Financieras Impulsada por IA Explicable usando Aprendizaje Automático

## Introducción


<p align = 'justify' >La Inteligencia Artificial (IA) ha trascendido la corriente principal de los negocios, convirtiéndose en el motor central para resolver los desafíos empresariales más complejos y de alto volumen. A medida que el mundo avanzaba en la transformación digital, el uso de los pagos en línea, las tarjetas, las billeteras digitales y las aplicaciones móviles se ha consolidado, reduciendo significativamente la dependencia del dinero en efectivo. </p>

<p align = 'justify' >La IA ya no es solo una herramienta de soporte; es esencial para la prevención de fraude en tiempo real, la personalización de la experiencia del cliente y la optimización de las tasas de aprobación de transacciones. </p>

<p align = 'justify' >Según las tendencias de mercado hasta 2025, Visa y Mastercard siguen siendo los líderes indiscutibles en el procesamiento de transacciones a nivel global, facilitando trillones de dólares en pagos y manteniendo una presencia casi universal en el comercio electrónico. Sin embargo, la competencia se ha intensificado con el crecimiento exponencial de: </p>

<p align = 'justify' >Pagos Instantáneos (Real-Time Payments): Sistemas impulsados por IA que validan y ejecutan transacciones en segundos, como el sistema Pix en Brasil o las iniciativas de pagos in-app y peer-to-peer (P2P). </p>

<p align = 'justify' >Billeteras Móviles: Plataformas como Apple Pay, Google Pay y Alipay/WeChat Pay (particularmente dominantes en Asia), que utilizan la IA para mejorar la seguridad a través de la biometría y el análisis de comportamiento. </p>

<p align = 'justify' >En regiones como el Reino Unido, Visa y Mastercard siguen siendo las opciones de pago fundamentales, disponibles en prácticamente el 100% de los principales minoristas y actuando como la infraestructura base sobre la cual se construyen las soluciones de pago más modernas. </p>



## 🏦 Impacto de la IA en las Finanzas y la Banca

La industria de las Finanzas y la Banca ha evolucionado de manera acelerada gracias a las tecnologías digitales, con la Inteligencia Artificial (IA) como su principal catalizador, dando forma a la disciplina de FinTech (Tecnología Financiera).

El impacto económico de la IA ya no es una mera proyección futura; es una realidad operativa. Si bien las estimaciones varían, la IA ha generado ahorros y eficiencias multimillonarias, superando ya las proyecciones iniciales. Se estima que el ahorro acumulado para la industria bancaria global debido a la implementación de la IA continuará su ascenso, con proyecciones a largo plazo que superan el billón de dólares para el final de la década, impulsado por la automatización de procesos y la reducción de pérdidas por fraude.

La IA es fundamental para crear soluciones avanzadas a los problemas tradicionales de los sectores financiero y bancario, con aplicaciones clave que se han vuelto estándar en la industria:

Detección de Fraude en Tiempo Real: Utilizando Machine Learning para analizar patrones de comportamiento de transacciones y prevenir pérdidas en milisegundos.

Cumplimiento Normativo (RegTech): Aplicación de IA para la Lucha contra el Blanqueo de Dinero (AML) y el conocimiento del cliente (KYC), automatizando el monitoreo de transacciones sospechosas.

Gestión de Riesgos: Análisis predictivo avanzado para el cálculo de la solvencia crediticia y la evaluación de riesgos de mercado.

Gestión de Inversiones (Robo-Advisors): Plataformas impulsadas por algoritmos para la creación automatizada de carteras, la reasignación de activos y la personalización de estrategias financieras.

Análisis Predictivo: Optimización de las operaciones, personalización de productos financieros y predicción de la deserción de clientes (churn).


<p align="center" width="100%">
<img alt="GIF" src="https://user-images.githubusercontent.com/31254745/191377492-9b827999-aba9-4dc7-8adf-fdb1b6c8fb19.png">
</p>

## Detección de Fraude en Transacciones Financieras

<p align = 'justify' >Si bien la digitalización crea oportunidades para el desarrollo y el crecimiento, también atrae a ciberdelincuentes y estafadores para el fraude financiero, que se ha convertido en un importante problema empresarial en la industria financiera y bancaria. </p>

<p align = 'justify' >Las pérdidas por fraude aumentaron en un 30% y los estafadores han robado 754 millones de libras esterlinas de las transacciones financieras bancarias y el 76% de las pérdidas por fraude de tarjetas de crédito en el Reino Unido se debieron a la modalidad de Tarjeta no Presente (CNP), por un total de 470,2 millones de libras esterlinas (UK Finance, 2021).</p>
<p align="center" width="100%">
<img alt="GIF" src="https://user-images.githubusercontent.com/31254745/191378636-97f1fe09-018e-4be3-a025-4a2330ded381.png">
</p>

## Problema de Investigación

### ¿Podemos 'confiar en la IA' solo porque es muy precisa?

<p align = 'justify' > En la detección de fraude financiero, se han aplicado varios métodos de aprendizaje automático para detectar comportamientos fraudulentos en los datos financieros. La mayoría de los sistemas actuales de detección de fraude se basan en modelos de caja negra, por lo que se vuelve más difícil entender y explicar las predicciones de estos sistemas a los responsables de la toma de decisiones empresariales o a los usuarios no expertos en IA.</p>

<p align = 'justify' > Este desafío de la “caja negra” es uno de los mayores obstáculos que impiden que los servicios financieros y la industria bancaria pongan en funcionamiento sus estrategias de IA en producción. Afortunadamente, la IA Explicable (XAI), una IA centrada en el ser humano, ayuda a aumentar la confianza, la transparencia y la confianza del usuario final al proporcionar explicaciones de los modelos de IA que son más comprensibles para los humanos para una mejor toma de decisiones empresariales.</p>

<p align = 'justify' >Este proyecto de investigación tiene como objetivo llenar el vacío de la falta de explicabilidad de los complejos modelos de IA de caja negra en la detección de fraude en transacciones financieras.</p>

## Objetivo y Objetivos de la Investigación

<p align = 'justify' >Este estudio de investigación tiene como objetivo implementar una “Interfaz impulsada por IA Explicable (XAI) y una Aplicación Web de Prueba de Concepto (POC) para la Detección de Fraude en Transacciones Financieras utilizando Aprendizaje Automático y Redes Neuronales Profundas” en la industria de Servicios Financieros y Banca. </p>

Para lograr este objetivo, se establecen los siguientes cuatro objetivos:

- **Objetivo 1:** Construir un motor de clasificación robusto para clasificar una transacción financiera como fraudulenta o legítima aplicando cinco algoritmos de aprendizaje automático y dos algoritmos de redes neuronales profundas.
- **Objetivo 2:** Evaluar el rendimiento de todos los resultados del modelo utilizando métricas como Precisión, Puntuación AUC-ROC, Matriz de Confusión, Recall, Precisión, Puntuación F1, Curva AUC-ROC y Curva Precisión-Recall.
- **Objetivo 3:** Implementar cinco métodos de IA Explicable (XAI) y una Interfaz de Explicabilidad para mejorar la confianza y la explicabilidad del modelo en los resultados del modelo con mejor rendimiento obtenidos en el objetivo 2.
- **Objetivo 4:** Desarrollar una Prueba de Concepto (POC) como una aplicación web de front-end para que los responsables de la toma de decisiones empresariales generen valor empresarial y realicen predicciones en tiempo real sobre la detección de fraudes.
