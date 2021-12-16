## Sartorius - Сегментация экземпляра ячейки
Обнаружение одиночных нейронных клеток на микроскопических изображениях

### Описание
Неврологические расстройства, включая нейродегенеративные заболевания, такие как болезнь Альцгеймера и опухоли головного мозга, являются ведущей причиной смерти и инвалидности во всем мире. Однако трудно количественно оценить, насколько хорошо эти смертельные заболевания поддаются лечению. Одним из общепринятых методов является изучение нейрональных клеток с помощью световой микроскопии, которая одновременно доступна и неинвазивна. К сожалению, сегментирование отдельных нейронных клеток на микроскопических изображениях может быть сложной задачей и требует много времени. Точная индивидуальная сегментация этих клеток - с помощью компьютерного зрения - может привести к новым и эффективным открытиям лекарств для лечения миллионов людей с этими заболеваниями.



Современные решения имеют ограниченную точность, в частности, для нейронных клеток. Во внутренних исследованиях по разработке моделей сегментации экземпляров клеток линия клеток нейробластомы SH-SY5Y стабильно демонстрирует самые низкие показатели точности из восьми различных протестированных типов раковых клеток. Это может быть связано с тем, что нейронные клетки имеют очень уникальную, нерегулярную и вогнутую морфологию, что затрудняет их сегментирование с помощью обычно используемых головок масок.

Sartorius является партнером биологических исследований и биофармацевтической промышленности. Они дают ученым и инженерам возможность упростить и ускорить прогресс в науках о жизни и биотехнологии, позволяя разрабатывать новые и более совершенные методы лечения и более доступные лекарства. Они являются магнитом и динамичной платформой для пионеров и ведущих экспертов в этой области. Они объединяют творческие умы для достижения общей цели: технологических достижений, ведущих к улучшению здоровья большего числа людей.

В этом соревновании вы обнаружите и очертите различные интересующие вас объекты на биологических изображениях, изображающих типы нервных клеток, обычно используемые при изучении неврологических расстройств. В частности, вы будете использовать изображения фазово-контрастной микроскопии для обучения и тестирования вашей модели, например, сегментации нейрональных клеток. Успешные модели сделают это с высокой степенью точности.

В случае успеха вы поможете дальнейшим исследованиям в области нейробиологии благодаря сбору надежных количественных данных. Исследователи могут использовать это для более простого измерения воздействия болезней и условий лечения на нейронные клетки. В результате могут быть открыты новые лекарства для лечения миллионов людей с этими основными причинами смерти и инвалидности.


### Оценка
Это соревнование оценивается по средней средней точности на разных пересечениях пороговых значений объединения (IoU). IoU предлагаемого набора объектных пикселей и набора истинных объектных пикселей рассчитывается как:


Показатель проходит через диапазон пороговых значений IoU, в каждой точке вычисляя среднее значение точности. Пороговые значения в диапазоне от 0,5 до 0,95 с размером шага 0,05: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95). Другими словами, при пороге 0,5 предсказанный объект считается "попаданием", если его пересечение по объединению с основным объектом истинности больше 0,5.

При каждом пороговом значении , значение точности вычисляется на основе количества истинных положительных результатов (TP), ложных отрицательных результатов (FN) и ложных срабатываний (FP), полученных в результате сравнения предсказанного объекта со всеми наземными объектами истинности:


Истинно положительный результат засчитывается, когда один предсказанный объект соответствует наземному объекту с IoU выше порогового значения. Ложное срабатывание указывает на то, что у предсказанного объекта не было связанного наземного объекта истинности. Ложноотрицательный результат указывает, что объект наземной истинности не имеет связанного с ним прогнозируемого объекта. Затем вычисляется средняя точность одного изображения как среднее из приведенных выше значений точности для каждого порогового значения IoU:


Наконец, оценка, возвращаемая метрикой конкуренции, представляет собой среднее значение, взятое из отдельных средних значений точности каждого изображения в тестовом наборе данных.

Представленный файл
Чтобы уменьшить размер отправляемого файла, наша метрика использует кодирование длин серий для значений пикселей. Вместо того, чтобы отправлять исчерпывающий список индексов для сегментации, вы отправляете пары значений, которые содержат начальную позицию и длину цикла. Например, «1 3» означает, что начинается с пикселя 1 и выполняется всего 3 пикселя (1,2,3).

Формат соревнования требует наличия списка пар, разделенных пробелами. Например, «1 3 10 5» означает, что пиксели 1,2,3,10,11,12,13,14 должны быть включены в маску. Пиксели индексируются по одному
и нумеруются сверху вниз, затем слева направо: 1 - пиксель (1,1), 2 - пиксель (2,1) и т. Д.

Метрика проверяет, что пары отсортированы, положительны и декодированные значения пикселей не дублируются. Он также проверяет, что никакие две предсказанные маски для одного и того же изображения не перекрываются.

Файл должен содержать заголовок и иметь следующий формат. Каждая строка в вашем представлении представляет собой одну прогнозируемую сегментацию ядра для данного ImageId.

### Требование к коду
Заявки на участие в этом конкурсе должны быть сделаны через Блокноты. Чтобы кнопка «Отправить» была активной после фиксации, должны быть выполнены следующие условия:

Ноутбук с процессором <= 9 часов работы
Ноутбук с графическим процессором <= 9 часов работы
Доступ в Интернет отключен
Разрешены свободно и общедоступные внешние данные, включая предварительно обученные модели.
Файл для отправки должен быть назван submission.csv
Пожалуйста, ознакомьтесь с часто задаваемыми вопросами о Code Competition для получения дополнительной информации о том, как подать заявку. И просмотрите документацию по отладке кода, если вы столкнулись с ошибками при отправке.

### Описание данных
В этом соревновании мы сегментируем нейронные клетки на изображениях. Обучающие аннотации представлены в виде масок с кодировкой длин серий, а изображения представлены в формате PNG. Количество изображений невелико, но количество аннотированных объектов достаточно велико. Набор скрытых тестов составляет примерно 240 изображений.

Примечание: хотя прогнозы не могут перекрываться, обучающие метки предоставляются полностью (с включенными перекрывающимися частями). Это необходимо для того, чтобы модели предоставляли полные данные для каждого объекта. Устранение дублирования прогнозов - задача конкурента.

