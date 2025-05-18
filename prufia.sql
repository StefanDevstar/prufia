/*
SQLyog Community v13.1.7 (64 bit)
MySQL - 10.4.32-MariaDB : Database - prufia
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`prufia` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci */;

USE `prufia`;

/*Table structure for table `scores` */

DROP TABLE IF EXISTS `scores`;

CREATE TABLE `scores` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `student_id` int(11) NOT NULL,
  `submission_id` int(11) NOT NULL,
  `confidence_score` decimal(5,2) NOT NULL,
  `result_json` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `semester_id` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `student_id` (`student_id`),
  KEY `submission_id` (`submission_id`),
  CONSTRAINT `scores_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `students` (`id`),
  CONSTRAINT `scores_ibfk_2` FOREIGN KEY (`submission_id`) REFERENCES `submissions` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `scores` */

/*Table structure for table `students` */

DROP TABLE IF EXISTS `students`;

CREATE TABLE `students` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name_or_alias` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `teacher_id` int(11) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `class` varchar(500) DEFAULT NULL,
  `semester_id` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `email` (`email`),
  KEY `fk_teacher` (`teacher_id`),
  CONSTRAINT `fk_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teachers` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `students` */

insert  into `students`(`id`,`name_or_alias`,`email`,`password_hash`,`teacher_id`,`created_at`,`class`,`semester_id`) values 
(9,'Bravo','bravo@gmail.com','a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3',2,'2025-05-09 13:31:19',NULL,'2025-Pilot'),
(10,'Alpha','alpha@gmail.com','a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3',1,'2025-05-09 13:32:10',NULL,'2025-Pilot');

/*Table structure for table `submissions` */

DROP TABLE IF EXISTS `submissions`;

CREATE TABLE `submissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `student_id` int(11) DEFAULT NULL,
  `baseline_1_path` varchar(500) DEFAULT NULL,
  `baseline_2_path` varchar(500) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `submission_path` varchar(500) DEFAULT NULL,
  `score_baseline_1` int(11) DEFAULT NULL,
  `score_baseline_2` int(11) DEFAULT NULL,
  `final_score` int(11) DEFAULT NULL,
  `trust_flag` varchar(500) DEFAULT NULL,
  `interpretation` varchar(500) DEFAULT NULL,
  `semester_id` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `student_id` (`student_id`),
  CONSTRAINT `submissions_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `students` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=48 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `submissions` */

insert  into `submissions`(`id`,`student_id`,`baseline_1_path`,`baseline_2_path`,`created_at`,`submission_path`,`score_baseline_1`,`score_baseline_2`,`final_score`,`trust_flag`,`interpretation`,`semester_id`) values 
(44,9,'baseline\\9\\60b3d66b-1415.txt','baseline\\9\\d012c31d-9252.txt','2025-05-15 17:17:59',NULL,NULL,NULL,NULL,NULL,NULL,'2025-Pilot'),
(45,10,'baseline\\10\\139adf0b-763a.txt','baseline\\10\\819cc73e-6508.txt','2025-05-15 17:34:20',NULL,NULL,NULL,NULL,NULL,NULL,'2025-Pilot'),
(46,9,'baseline\\9\\94b4e6b9-9feb.txt','baseline\\9\\c4074080-c273.txt','2025-05-15 19:07:31',NULL,NULL,NULL,NULL,NULL,NULL,'2025-Pilot'),
(47,9,'baseline\\9\\afc9c2bc-b4c3.txt','baseline\\9\\4dddd82e-b529.txt','2025-05-15 19:20:49',NULL,NULL,NULL,NULL,NULL,NULL,'2025-Pilot');

/*Table structure for table `teachers` */

DROP TABLE IF EXISTS `teachers`;

CREATE TABLE `teachers` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `email` varchar(255) DEFAULT NULL,
  `password_hash` varchar(255) DEFAULT NULL,
  `role` enum('teacher','admin') DEFAULT 'teacher',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `semester_id` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `teachers` */

insert  into `teachers`(`id`,`name`,`email`,`password_hash`,`role`,`created_at`,`semester_id`) values 
(1,'Teacher Arnold','anold@gmail.com','sdfsdfsd','teacher','2025-05-09 13:25:10','2025-Pilot'),
(2,'Teacher Roland','roland@aeo.com','4345t43fdf','teacher','2025-05-09 13:26:58','2025-Pilot');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
