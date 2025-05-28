/*
SQLyog Trial v13.1.8 (64 bit)
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

/*Table structure for table `class` */

DROP TABLE IF EXISTS `class`;

CREATE TABLE `class` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `label` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `class` */

insert  into `class`(`id`,`label`) values 
(1,'math'),
(2,'computer');

/*Table structure for table `logs` */

DROP TABLE IF EXISTS `logs`;

CREATE TABLE `logs` (
  `id` bigint(255) DEFAULT NULL,
  `ip` varchar(255) DEFAULT NULL,
  `connectedTime` datetime DEFAULT NULL,
  `mac` varchar(500) DEFAULT NULL,
  `action` varchar(500) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `logs` */

/*Table structure for table `passcode` */

DROP TABLE IF EXISTS `passcode`;

CREATE TABLE `passcode` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `stdId` int(11) NOT NULL,
  `passcode` varchar(255) NOT NULL,
  `used` tinyint(4) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `passcode` */

insert  into `passcode`(`id`,`stdId`,`passcode`,`used`,`created_at`) values 
(1,9,'KxNiqodPO',0,'2025-05-27 18:15:22'),
(2,10,'N0SnAJsXO',1,'2025-05-27 09:24:28');

/*Table structure for table `resubmit_request` */

DROP TABLE IF EXISTS `resubmit_request`;

CREATE TABLE `resubmit_request` (
  `id` bigint(255) NOT NULL AUTO_INCREMENT,
  `base_id` bigint(255) DEFAULT NULL,
  `feedback` varchar(500) DEFAULT NULL,
  `teacherid` bigint(255) DEFAULT NULL,
  `semesterid` bigint(255) DEFAULT NULL,
  `status` int(1) NOT NULL,
  `created_at` datetime NOT NULL,
  `approved_at` datetime DEFAULT NULL,
  KEY `id` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `resubmit_request` */

insert  into `resubmit_request`(`id`,`base_id`,`feedback`,`teacherid`,`semesterid`,`status`,`created_at`,`approved_at`) values 
(10,84,'short description',NULL,NULL,1,'2025-05-27 11:16:07',NULL),
(11,85,'test',NULL,NULL,1,'2025-05-28 05:30:21',NULL);

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
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `scores` */

/*Table structure for table `semesters` */

DROP TABLE IF EXISTS `semesters`;

CREATE TABLE `semesters` (
  `id` bigint(255) NOT NULL AUTO_INCREMENT,
  `label` varchar(500) DEFAULT NULL,
  `startDate` date DEFAULT NULL,
  `endDate` date DEFAULT NULL,
  KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `semesters` */

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
  `class_id` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `students` */

insert  into `students`(`id`,`name_or_alias`,`email`,`password_hash`,`teacher_id`,`created_at`,`class`,`semester_id`,`class_id`) values 
(9,'Bravo','bravo@gmail.com','a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3',2,'2025-05-09 13:31:19',NULL,'2025-Pilot',1),
(10,'Alpha','alpha@gmail.com','a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3',1,'2025-05-09 13:32:10',NULL,'2025-Pilot',2);

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
  `ip` varchar(500) NOT NULL,
  `hash_signature` varchar(255) NOT NULL,
  `salt` binary(255) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `student_id` (`student_id`),
  CONSTRAINT `submissions_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `students` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=86 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

/*Data for the table `submissions` */

insert  into `submissions`(`id`,`student_id`,`baseline_1_path`,`baseline_2_path`,`created_at`,`submission_path`,`score_baseline_1`,`score_baseline_2`,`final_score`,`trust_flag`,`interpretation`,`semester_id`,`ip`,`hash_signature`,`salt`) values 
(84,9,'C:\\Users\\Administrator\\Documents\\soumya\\prufia\\baseline\\9\\baseline1_6345bce7-8c89-4db6-9daf-5daa012bfa1d.enc','C:\\Users\\Administrator\\Documents\\soumya\\prufia\\baseline\\9\\baseline2_6345bce7-8c89-4db6-9daf-5daa012bfa1d.enc','2025-05-27 11:01:20',NULL,NULL,NULL,NULL,NULL,NULL,'2025-Pilot','83.234.227.51','','xS®/‰∑ı–_4ÍEò‰r\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0'),
(85,10,'C:\\Users\\Administrator\\Documents\\soumya\\prufia\\baseline\\10\\baseline1_ac01714d-f439-4d8b-b244-54ea94a0aa19.enc','C:\\Users\\Administrator\\Documents\\soumya\\prufia\\baseline\\10\\baseline2_ac01714d-f439-4d8b-b244-54ea94a0aa19.enc','2025-05-28 02:22:57',NULL,NULL,NULL,NULL,NULL,NULL,'2025-Pilot','83.234.227.51','','“¢√2b˙Üè‚ê=§=ü˚ë\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0');

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
