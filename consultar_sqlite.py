import sqlite3

def obtener_informacion_alumnos(nombre_alumno=None):
    conn = sqlite3.connect('alumnos.db')
    c = conn.cursor()

    if nombre_alumno is None:
        # Obtener información de todos los alumnos
        c.execute("SELECT * FROM Alumnos")
        alumnos = c.fetchall()
    else:
        # Obtener información del alumno
        c.execute("SELECT * FROM Alumnos WHERE Nombre=?", (nombre_alumno,))
        alumnos = [c.fetchone()]

    informacion_alumnos = []
    for alumno in alumnos:
        if alumno is None:
            print("No se encontró al alumno")
            continue

        # Obtener las notas del alumno
        c.execute("SELECT * FROM Notas WHERE AlumnoID=?", (alumno[0],))
        notas = c.fetchall()

        # Obtener las asistencias del alumno
        c.execute("SELECT * FROM Asistencias WHERE AlumnoID=?", (alumno[0],))
        asistencias = c.fetchall()

        # Obtener las materias del alumno
        c.execute("SELECT * FROM Materias WHERE ID IN (SELECT MateriaID FROM Notas WHERE AlumnoID=?)", (alumno[0],))
        materias = c.fetchall()

        informacion_alumnos.append((alumno, notas, asistencias, materias))

    conn.close()

    return informacion_alumnos

if __name__ == "__main__":
    nombre_alumno = input("Ingrese el nombre del alumno (deje en blanco para todos los alumnos): ")
    informacion_alumnos = obtener_informacion_alumnos(nombre_alumno if nombre_alumno != '' else None)
    for alumno, notas, asistencias, materias in informacion_alumnos:
        print("Información del alumno:", alumno)
        print("Notas:", notas)
        print("Asistencias:", asistencias)
        print("Materias:", materias)
        print()