package io.github.mrmathami;

import nu.pattern.OpenCV;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

public class Main {
	public static @Nullable Double calculate(@NotNull Mat inputImage, @NotNull Scalar color,
			double precisionPercentage, @NotNull Mat debugImage) {
		// masking
		final Mat mask = new Mat();
		Core.inRange(inputImage, Scalar.all(1 - precisionPercentage).mul(color),
				Scalar.all(1 + precisionPercentage).mul(color), mask);

		// finding contours
		final List<MatOfPoint> contours = new LinkedList<>();
		final Mat hierarchy = new Mat();
		Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

		// draw contours with red
		Imgproc.drawContours(debugImage, contours, -1, new Scalar(0, 0, 255), 1);

		// remove all tiny contours
		for (final Iterator<MatOfPoint> iterator = contours.iterator(); iterator.hasNext(); ) {
			final MatOfPoint contour = iterator.next();
			final Rect boundingRect = Imgproc.boundingRect(contour);
			if (boundingRect.width < 30 || boundingRect.height < 40) {
				iterator.remove();
			}
		}

		// draw filtered contours with green
//		System.out.println("contours = " + contours);
		Imgproc.drawContours(debugImage, contours, -1, new Scalar(0, 255, 0), 1);

		// check if contour list has only 2 points
		if (contours.size() != 2) return null;

		// find center points
		final List<Point> centerPoints = new ArrayList<>();
		for (final MatOfPoint contour : contours) {
			// convert to Hull points
			final MatOfInt pointIndexes = new MatOfInt();
			Imgproc.convexHull(contour, pointIndexes, true);
			final List<Point> hullPoints = Arrays.stream(pointIndexes.toArray())
					.mapToObj(contour.toList()::get)
					.toList();
			// draw Hull points with purple
			Imgproc.polylines(debugImage, List.of(new MatOfPoint(hullPoints.toArray(Point[]::new))), true, new Scalar(0, 255, 255), 1);
			// find parallelogram
			final List<Point> parallelogram = mep(hullPoints);
			// draw parallelogram with blue
			Imgproc.polylines(debugImage, List.of(new MatOfPoint(parallelogram.toArray(Point[]::new))), true, new Scalar(255, 0, 0), 1);
			// calculate center point
			final List<Point> points = parallelogram.stream().sorted(Comparator.comparingDouble(value -> value.y)).toList();
			final Point topPointA = points.get(0);
			final Point topPointB = points.get(1);
			final Point centerPoint = new Point((topPointA.x + topPointB.x) / 2.0, (topPointA.y + topPointB.y) / 2.0);
			Imgproc.circle(debugImage, centerPoint, 5, new Scalar(255, 0, 0), 1);
			centerPoints.add(centerPoint);
		}
		final Point pointA = centerPoints.get(0);
		final Point pointB = centerPoints.get(1);
		Imgproc.line(debugImage, pointA, pointB, new Scalar(255, 0, 0), 1);
		return Math.signum(pointA.x - pointB.x) * Math.signum(pointA.y - pointB.y) * euclideanDist(pointA, pointB);
	}

	public static void main(@NotNull String @NotNull [] args) throws Exception {
		final String adbPath = args.length >= 1 ? args[0] : "/usr/bin/adb";
		final double rightMultiplier = args.length >= 2 ? Double.parseDouble(args[1]) : -1.65;
		final double leftMultiplier = args.length >= 3 ? Double.parseDouble(args[2]) : 1.35;
		final long delayMillisecondsAfterJump = args.length >= 4 ? Long.parseLong(args[3]) : 3000;
		System.out.println("adb (default \"/usr/bin/adb\"): " + adbPath);
		System.out.println("rightMultiplier (default -1.65): " + rightMultiplier);
		System.out.println("leftMultiplier (default 1.35): " + leftMultiplier);
		System.out.println("delayMillisecondsAfterJump (default 3000): " + delayMillisecondsAfterJump);
		OpenCV.loadLocally();

		try (final PrintWriter writer = new PrintWriter(Files.newBufferedWriter(Path.of("output.csv"),
				StandardCharsets.UTF_8, StandardOpenOption.CREATE,
				StandardOpenOption.WRITE, StandardOpenOption.APPEND));
				final Scanner scanner = new Scanner(System.in)) {
			System.out.print("Press Enter to continue.");
			scanner.nextLine();
			writer.println();
			int failed = 0;
			while (true) {
				final long time = System.currentTimeMillis();

				// capture
				final Path inputPath = Path.of("screen-" + time + ".png");
				screenCapture(adbPath, inputPath);

				// load
				final Mat inputImage = loadImage(inputPath);
				final Mat croppedImage = new Mat(inputImage, new Range(1000, 2080), new Range(0, 1080));
				final Mat debugImage = croppedImage.clone();

				//masking image
				final Double distanceA = calculate(croppedImage, new Scalar(0x95, 0xBC, 0xFF), 0.02, debugImage);
				final Double distanceB = calculate(croppedImage, new Scalar(0x8A, 0xB1, 0xF8), 0.02, debugImage);


				// jump
				if (distanceA != null && distanceB != null) {
					System.out.println("Jumping...");
					double distance = distanceA + distanceB;
					// left: positive
					// right: negative
					long holdTime = Math.round(distance * (distance < 0 ? rightMultiplier : leftMultiplier));
					jump(adbPath, holdTime);

//					// is that jump good?
//					System.out.print("Input result: ");
//					final String resultLine = scanner.nextLine();
//					if (!resultLine.matches("\\d+")) break;
					writer.printf("\"%s\",%s,%s,%s,%s\n", inputPath, holdTime, distanceA, distanceB, distance);

					// save debug image
					final Path debugPath = Path.of("screen-" + time + "-" + distance + "-debug.png");
					saveImage(debugImage, debugPath);

					Thread.sleep(delayMillisecondsAfterJump);
					failed = 0;
				} else {
					System.out.println("Failed to jump!");
					Files.deleteIfExists(inputPath);
					failed++;
					if (failed >= 3) {
						if (failed >= 4) break;
						restart(adbPath);
						Thread.sleep(5000);
					} else {
						Thread.sleep(500);
					}
				}
			}
		}
	}

	private static void screenCapture(@NotNull String adbPath, @NotNull Path outputFile) throws IOException, InterruptedException {
		final Process process = new ProcessBuilder(adbPath, "exec-out", "screencap", "-p").start();
		try (final OutputStream outputStream = Files.newOutputStream(outputFile)) {
			process.getInputStream().transferTo(outputStream);
		}
		process.waitFor();
	}

	private static void jump(@NotNull String adbPath, long holdTime) throws IOException, InterruptedException {
		new ProcessBuilder(adbPath, "shell", "input", "touchscreen", "swipe",
				"600", "600", "600", "600", Long.toString(holdTime)).start().waitFor();
	}

	private static void restart(@NotNull String adbPath) throws IOException, InterruptedException {
		new ProcessBuilder(adbPath, "shell", "input", "touchscreen", "tap",
				"750", "1550").start().waitFor();
	}

	private static double euclideanDist(@NotNull Point p, @NotNull Point q) {
		return Math.sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
	}

	public static @NotNull Mat loadImage(@NotNull Path imagePath) {
		return Imgcodecs.imread(imagePath.toString());
	}

	public static boolean saveImage(@NotNull Mat imageMatrix, @NotNull Path targetPath) {
		return Imgcodecs.imwrite(targetPath.toString(), imageMatrix);
	}


	//region Minimal Enclosing Parallelogram
	// https://stackoverflow.com/questions/38409156/minimal-enclosing-parallelogram-in-python

	private static double distance(@NotNull Point p1, @NotNull Point p2, @NotNull Point p) {
		return Math.abs(((p2.y - p1.y) * p.x - (p2.x - p1.x) * p.y + p2.x * p1.y - p2.y * p1.x) /
				Math.sqrt((p2.y - p1.y) * (p2.y - p1.y) + (p2.x - p1.x) * (p2.x - p1.x)));
	}

	private static @NotNull List<@NotNull Integer> antipodal_pairs(@NotNull List<@NotNull Point> convex_polygon) {
		final List<Integer> l = new ArrayList<>();
		final int n = convex_polygon.size();
		Point p1 = convex_polygon.get(0);
		Point p2 = convex_polygon.get(1);

		int t = 0;
		double d_max = 0;
		for (int p = 1; p < n; p++) {
			double d = distance(p1, p2, convex_polygon.get(p));
			if (d > d_max) {
				t = p;
				d_max = d;
			}
		}
		l.add(t);

		for (int p = 1; p < n; p++) {
			p1 = convex_polygon.get(p % n);
			p2 = convex_polygon.get((p + 1) % n);
			Point _p = convex_polygon.get(t % n);
			Point _pp = convex_polygon.get((t + 1) % n);
			while (distance(p1, p2, _pp) > distance(p1, p2, _p)) {
				t = (t + 1) % n;
				_p = convex_polygon.get(t % n);
				_pp = convex_polygon.get((t + 1) % n);
			}
			l.add(t);
		}
		return l;
	}

	private static @NotNull Point parallel_vector(@NotNull Point a, @NotNull Point b, @NotNull Point c) {
		Point v0 = new Point(c.x - a.x, c.y - a.y);
		Point v1 = new Point(b.x - c.x, b.y - c.y);
		return new Point(c.x - v0.x - v1.x, c.y - v0.y - v1.y);
	}

	private static @NotNull Point line_intersection(Point p1, Point p2, Point p3, Point p4) {
		final double a = p1.x * p2.y - p1.y * p2.x;
		final double b = p3.x * p4.y - p3.y * p4.x;
		final double c = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
		double px = (a * (p3.x - p4.x) - (p1.x - p2.x) * b) / c;
		double py = (a * (p3.y - p4.y) - (p1.y - p2.y) * b) / c;
		return new Point(px, py);
	}

	private static Map.Entry<Double, List<Point>> compute_parallelogram(@NotNull List<@NotNull Point> convex_polygon,
			@NotNull List<@NotNull Integer> l, int z1, int z2) {
		final int n = convex_polygon.size();
		Point p1 = convex_polygon.get(z1 % n);
		Point p2 = convex_polygon.get((z1 + 1) % n);
		Point q1 = convex_polygon.get(z2 % n);
		Point q2 = convex_polygon.get((z2 + 1) % n);
		Point ap1 = convex_polygon.get(l.get(z1 % n));
		Point aq1 = convex_polygon.get(l.get(z2 % n));
		Point ap2 = parallel_vector(p1, p2, ap1);
		Point aq2 = parallel_vector(q1, q2, aq1);
		Point a = line_intersection(p1, p2, q1, q2);
		Point b = line_intersection(p1, p2, aq1, aq2);
		Point d = line_intersection(ap1, ap2, q1, q2);
		Point c = line_intersection(ap1, ap2, aq1, aq2);
		double s = distance(a, b, c) * Math.sqrt((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
		return Map.entry(s, List.of(a, b, c, d));
	}

	private static List<Point> mep(@NotNull List<@NotNull Point> convex_polygon) {
		int z1 = 0;
		int z2 = 0;
		final int n = convex_polygon.size();
		List<Integer> l = antipodal_pairs(convex_polygon);

		double so = Double.MAX_VALUE;
		List<Point> abcdo = null;
		int z1o;
		int z2o;

		for (z1 = 0; z1 < n; z1++) {
			if (z1 > z2) z2 = z1 + 1;
			Point p1 = convex_polygon.get(z1 % n);
			Point p2 = convex_polygon.get((z1 + 1) % n);
			Point a = convex_polygon.get(z2 % n);
			Point b = convex_polygon.get((z2 + 1) % n);
			Point c = convex_polygon.get(l.get(z2 % n));
			if (distance(p1, p2, a) >= distance(p1, p2, b)) continue;

			while (distance(p1, p2, c) > distance(p1, p2, b)) {
				z2 += 1;
				a = convex_polygon.get(z2 % n);
				b = convex_polygon.get((z2 + 1) % n);
				c = convex_polygon.get(l.get(z2 % n));
			}

			final Map.Entry<Double, List<Point>> t = compute_parallelogram(convex_polygon, l, z1, z2);
			double st = t.getKey();
			List<Point> abcdt = t.getValue();

			if (st < so) {
				so = st;
				abcdo = abcdt;
				z1o = z1;
				z2o = z2;
			}
		}

		return abcdo;
	}
/*

def distance(p1, p2, p):
    return abs(((p2[1]-p1[1])*p[0] - (p2[0]-p1[0])*p[1] + p2[0]*p1[1] - p2[1]*p1[0]) /
        math.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2))

def antipodal_pairs(convex_polygon):
    l = []
    n = len(convex_polygon)
    p1, p2 = convex_polygon[0], convex_polygon[1]

    t, d_max = None, 0
    for p in range(1, n):
        d = distance(p1, p2, convex_polygon[p])
        if d > d_max:
            t, d_max = p, d
    l.append(t)

    for p in range(1, n):
        p1, p2 = convex_polygon[p % n], convex_polygon[(p+1) % n]
        _p, _pp = convex_polygon[t % n], convex_polygon[(t+1) % n]
        while distance(p1, p2, _pp) > distance(p1, p2, _p):
            t = (t + 1) % n
            _p, _pp = convex_polygon[t % n], convex_polygon[(t+1) % n]
        l.append(t)

    return l


# returns score, area, points from top-left, clockwise , favouring low area
def mep(convex_polygon):
    def compute_parallelogram(convex_polygon, l, z1, z2):
        def parallel_vector(a, b, c):
            v0 = [c[0]-a[0], c[1]-a[1]]
            v1 = [b[0]-c[0], b[1]-c[1]]
            return [c[0]-v0[0]-v1[0], c[1]-v0[1]-v1[1]]

        # finds intersection between lines, given 2 points on each line.
        # (x1, y1), (x2, y2) on 1st line, (x3, y3), (x4, y4) on 2nd line.
        def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))/((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
            return px, py


        # from each antipodal point, draw a parallel vector,
        # so ap1->ap2 is parallel to p1->p2
        #    aq1->aq2 is parallel to q1->q2
        p1, p2 = convex_polygon[z1 % n], convex_polygon[(z1+1) % n]
        q1, q2 = convex_polygon[z2 % n], convex_polygon[(z2+1) % n]
        ap1, aq1 = convex_polygon[l[z1 % n]], convex_polygon[l[z2 % n]]
        ap2, aq2 = parallel_vector(p1, p2, ap1), parallel_vector(q1, q2, aq1)

        a = line_intersection(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1], q2[0], q2[1])
        b = line_intersection(p1[0], p1[1], p2[0], p2[1], aq1[0], aq1[1], aq2[0], aq2[1])
        d = line_intersection(ap1[0], ap1[1], ap2[0], ap2[1], q1[0], q1[1], q2[0], q2[1])
        c = line_intersection(ap1[0], ap1[1], ap2[0], ap2[1], aq1[0], aq1[1], aq2[0], aq2[1])

        s = distance(a, b, c) * math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
        return s, a, b, c, d


    z1, z2 = 0, 0
    n = len(convex_polygon)

    # for each edge, find antipodal vertice for it (step 1 in paper).
    l = antipodal_pairs(convex_polygon)

    so, ao, bo, co, do, z1o, z2o = 100000000000, None, None, None, None, None, None

    # step 2 in paper.
    for z1 in range(0, n):
        if z1 >= z2:
            z2 = z1 + 1
        p1, p2 = convex_polygon[z1 % n], convex_polygon[(z1+1) % n]
        a, b, c = convex_polygon[z2 % n], convex_polygon[(z2+1) % n], convex_polygon[l[z2 % n]]
        if distance(p1, p2, a) >= distance(p1, p2, b):
            continue

        while distance(p1, p2, c) > distance(p1, p2, b):
            z2 += 1
            a, b, c = convex_polygon[z2 % n], convex_polygon[(z2+1) % n], convex_polygon[l[z2 % n]]

        st, at, bt, ct, dt = compute_parallelogram(convex_polygon, l, z1, z2)

        if st < so:
            so, ao, bo, co, do, z1o, z2o = st, at, bt, ct, dt, z1, z2

    return so, ao, bo, co, do, z1o, z2o
*/

	//endregion Minimal Enclosing Parallelogram
}