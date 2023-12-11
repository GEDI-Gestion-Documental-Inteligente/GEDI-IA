import sqlite3
from datetime import date, timedelta
import random
from faker import Faker

def main():
    fake = Faker()
    conn = sqlite3.connect('alumnos.db')
    c = conn.cursor()

    # Eliminar tablas si existen
    c.execute('DROP TABLE IF EXISTS Alumnos')
    c.execute('DROP TABLE IF EXISTS Materias')
    c.execute('DROP TABLE IF EXISTS Asistencias')
    c.execute('DROP TABLE IF EXISTS Notas')
    c.execute('DROP TABLE IF EXISTS Profesores')
    c.execute('DROP TABLE IF EXISTS Exámenes')

    # Crear tablas
    c.execute('''
        CREATE TABLE Alumnos (
            ID INTEGER PRIMARY KEY,
            Nombre TEXT NOT NULL,
            Carrera TEXT NOT NULL,
            Correo TEXT NOT NULL,
            Telefono TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE Materias (
            ID INTEGER PRIMARY KEY,
            Nombre TEXT NOT NULL,
            ProfesorID INTEGER,
            FOREIGN KEY (ProfesorID) REFERENCES Profesores (ID)
        )
    ''')

    c.execute('''
        CREATE TABLE Asistencias (
            ID INTEGER PRIMARY KEY,
            AlumnoID INTEGER,
            MateriaID INTEGER,
            Fecha DATE,
            Presente INTEGER,
            FOREIGN KEY (AlumnoID) REFERENCES Alumnos (ID),
            FOREIGN KEY (MateriaID) REFERENCES Materias (ID)
        )
    ''')

    c.execute('''
        CREATE TABLE Notas (
            ID INTEGER PRIMARY KEY,
            AlumnoID INTEGER,
            MateriaID INTEGER,
            ExamenID INTEGER,
            Nota INTEGER,
            FOREIGN KEY (AlumnoID) REFERENCES Alumnos (ID),
            FOREIGN KEY (MateriaID) REFERENCES Materias (ID),
            FOREIGN KEY (ExamenID) REFERENCES Exámenes (ID)
        )
    ''')

    c.execute('''
        CREATE TABLE Profesores (
            ID INTEGER PRIMARY KEY,
            Nombre TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE Exámenes (
            ID INTEGER PRIMARY KEY,
            Nombre TEXT NOT NULL,
            MateriaID INTEGER,
            AlumnoID INTEGER,
            Fecha DATE,
            FOREIGN KEY (MateriaID) REFERENCES Materias (ID),
            FOREIGN KEY (AlumnoID) REFERENCES Alumnos (ID)
        )
    ''')

    # Agregar carreras
    carreras = ['Desarrollo de Software', 'Mecatronica']

    # Agregar alumnos con nombres aleatorios
    alumnos = [(fake.name(), carreras[i % 2], fake.email(), fake.phone_number()) for i in range(10)]
    c.executemany('INSERT INTO Alumnos (Nombre, Carrera, Correo, Telefono) VALUES (?,?,?,?)', alumnos)

    # Agregar profesores
    profesores = [(fake.name(),) for _ in range(5)]
    c.executemany('INSERT INTO Profesores (Nombre) VALUES (?)', profesores)

    # Agregar materias
    materias = [('Matematicas', 1),
                ('Algoritmo', 2),
                ('Base de Datos', 3),
                ('Programacion', 4),
                ('Ingenieria de Software', 5)]

    c.executemany('INSERT INTO Materias (Nombre, ProfesorID) VALUES (?,?)', materias)

    # Agregar exámenes
    for i in range(1, 11):  # Para cada alumno
        for j in range(1, 6):  # Para cada materia
            fecha = date.today() - timedelta(days=random.randint(1, 30))  # Fecha aleatoria en los últimos 30 días
            examen = ('Examen ' + str(j), j, i, fecha)
            c.execute('INSERT INTO Exámenes (Nombre, MateriaID, AlumnoID, Fecha) VALUES (?,?,?,?)', examen)

    # Agregar notas y asistencias para cada alumno en cada materia
    for i in range(1, 11):  # Para cada alumno
        for j in range(1, 6):  # Para cada materia
            # Nota
            for k in range(1, 6):  # Para cada examen
                nota = (i, j, k, 80 + i)  # Nota entre 80 y 90
                c.execute('INSERT INTO Notas (AlumnoID, MateriaID, ExamenID, Nota) VALUES (?,?,?,?)', nota)

            # Asistencias de los últimos 15 días
            for k in range(15):
                fecha = date.today() - timedelta(days=k)
                asistencia = random.randint(0, 1)  # Asistencia aleatoria
                registro = (i, j, fecha, asistencia)
                c.execute('INSERT INTO Asistencias (AlumnoID, MateriaID, Fecha, Presente) VALUES (?,?,?,?)', registro)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()