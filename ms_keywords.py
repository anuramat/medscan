sections2kws = {
"anamnesis": ["другой анамнез", "анамнез", "из анамнеза", "анамнез жизни","анамнез заболевания","анамнез болезни","в анамнезе"],
"complaints": ["жалобы", "жалобы при поступлении"],
"diagnosis": ["диагноз", "диагноз при выписке", "диагноз при поступлении"],
"state": ["состояние", "состояние при поступлении в стационар", "при поступлении", "при выписке", "состояние при выписке", "состояние при поступлении"],
"assignments": ["назначения"],
"recommendation": ["рекомендации", "рекомендовано"],
"comments": ["комментарий", "комментарии", "заключение", "комментарии и динамика состояния"],
"junk": ["junk"]
}

for section in sections2kws:
    for idx in range(len(sections2kws[section])):
        sections2kws[section][idx] = sections2kws[section][idx][0].upper() + sections2kws[section][idx][1:]

fields2kws = {
"name": ['Пациент', 'Ф.И.О. больного', 'Больной', 'ФИО', 'Больной (ая)', 'Ф.И.О. пациента', 'Пациент', 'Ф.И.О.', 'Больной (-ая)', 'Пациентка', 'Больная'],
"date": []
}
