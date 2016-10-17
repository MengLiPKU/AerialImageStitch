
#include "exposure_compensator.h"



void MyExposureCompensator::createWeightMaps( const vector<Point> &corners, const vector<Mat> &images, 
	const vector<Mat> &masks, vector<Mat_<float>> &ec_maps )
{
	vector<pair<Mat,uchar> > level_masks;
	for (size_t i = 0; i < masks.size(); ++i)
		level_masks.push_back(make_pair(masks[i], 255));
	createWeightMaps(corners, images, level_masks, ec_maps);
}


void MyExposureCompensator::createWeightMaps( const vector<Point> &corners, const vector<Mat> &images, 
	const vector<pair<Mat,uchar>> &masks, vector<Mat_<float>> &ec_maps )
{
	CV_Assert(corners.size() == images.size() && images.size() == masks.size());

	const int num_images = static_cast<int>(images.size());

	vector<Size> bl_per_imgs(num_images);
	vector<Point> block_corners;
	vector<Mat> block_images;
	vector<pair<Mat,uchar> > block_masks;

	// Construct blocks for gain compensator
	//	第一张图只取同其他图有重叠的block
	Size bl_per_img0((images[0].cols + bl_width_ - 1) / bl_width_, (images[0].rows + bl_height_ - 1) / bl_height_);
	bl_per_imgs[0] = bl_per_img0;
	Mat_<bool> block_mask0 = Mat_<bool>::zeros(bl_per_img0);
	int bl_width0 = (images[0].cols + bl_per_img0.width - 1) / bl_per_img0.width;
	int bl_height0 = (images[0].rows + bl_per_img0.height - 1) / bl_per_img0.height;
	for (int img_idx = 1; img_idx < num_images; ++img_idx)
	{
		Point corner = corners[img_idx] - corners[0];
		int left = max(0, corner.x);
		int right = min(images[0].cols, corner.x + images[img_idx].cols) - 1;
		int up = max(0, corner.y);
		int down = min(images[0].rows, corner.y + images[img_idx].rows) - 1;
		if (left >= right || up >= down)
			continue;
		for (int by = up / bl_height0; by < (down + bl_height0 - 1) / bl_height0; ++by)
			for (int bx = left / bl_width0; bx < (right + bl_width0 - 1) / bl_width0; ++bx)
				block_mask0(by, bx) = true;
	}
	for (int by = 0; by < bl_per_img0.height; ++by)
	{
		for (int bx = 0; bx < bl_per_img0.width; ++bx)
		{
			if (! block_mask0(by, bx))
				continue;
			Point bl_tl(bx * bl_width0, by * bl_height0);
			Point bl_br(min(bl_tl.x + bl_width0, images[0].cols),
				min(bl_tl.y + bl_height0, images[0].rows));

			block_corners.push_back(corners[0] + bl_tl);
			block_images.push_back(images[0](Rect(bl_tl, bl_br)));
			block_masks.push_back(make_pair(masks[0].first(Rect(bl_tl, bl_br)), masks[0].second));
		}
	}

	//	其他的图全部的block都计算
	for (int img_idx = 1; img_idx < num_images; ++img_idx)
	{
		Size bl_per_img((images[img_idx].cols + bl_width_ - 1) / bl_width_,
			(images[img_idx].rows + bl_height_ - 1) / bl_height_);
		int bl_width = (images[img_idx].cols + bl_per_img.width - 1) / bl_per_img.width;
		int bl_height = (images[img_idx].rows + bl_per_img.height - 1) / bl_per_img.height;
		bl_per_imgs[img_idx] = bl_per_img;
		for (int by = 0; by < bl_per_img.height; ++by)
		{
			for (int bx = 0; bx < bl_per_img.width; ++bx)
			{
				Point bl_tl(bx * bl_width, by * bl_height);
				Point bl_br(min(bl_tl.x + bl_width, images[img_idx].cols),
					min(bl_tl.y + bl_height, images[img_idx].rows));

				block_corners.push_back(corners[img_idx] + bl_tl);
				block_images.push_back(images[img_idx](Rect(bl_tl, bl_br)));
				block_masks.push_back(make_pair(masks[img_idx].first(Rect(bl_tl, bl_br)), masks[img_idx].second));
			}
		}
	}

	//	曝光补偿
	GainCompensator compensator;
	compensator.feed(block_corners, block_images, block_masks);
	vector<double> gains = compensator.gains();
	ec_maps.resize(num_images);

	Mat_<float> ker(1, 3);
	ker(0,0) = 0.25; ker(0,1) = 0.5; ker(0,2) = 0.25;

	int bl_idx = 0;
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		Size bl_per_img = bl_per_imgs[img_idx];
		ec_maps[img_idx].create(bl_per_img);

		for (int by = 0; by < bl_per_img.height; ++by)
		{
			for (int bx = 0; bx < bl_per_img.width; ++bx)
			{
				if ((img_idx == 0 && block_mask0(by, bx)) || img_idx != 0)
					ec_maps[img_idx](by, bx) = static_cast<float>(gains[bl_idx++]);
				else
					ec_maps[img_idx](by, bx) = 1.0;
			}
		}
		sepFilter2D(ec_maps[img_idx], ec_maps[img_idx], CV_32F, ker, ker);
		sepFilter2D(ec_maps[img_idx], ec_maps[img_idx], CV_32F, ker, ker);
	}

	double max_ec = 1.0f;
	double max_ec_i, min_ec_i;
	int i = 0;
	//for (int i = 0; i < num_images; i++)
	//{
	//	cv::minMaxIdx(ec_maps[i], &min_ec_i, &max_ec_i);
	//	max_ec = std::max(max_ec, max_ec_i);
	//}
	for (int i = 0; i < num_images; i++)
		ec_maps[i] = ec_maps[i] / ((float)(max_ec));

	//	将权重图resize到原始大小
	for(int i = 0; i < num_images; i++)
	{
		Mat_<float> gain_map;
		resize(ec_maps[i], gain_map, images[i].size(), 0, 0, INTER_LINEAR);
		ec_maps[i] = gain_map.clone();
	}

	//	将其他图的权重按照第一张图归一化，使得第一张图的权重全部是1
	for (int img_idx = 1; img_idx < num_images; ++img_idx)
	{
		Point corner = corners[0] - corners[img_idx];
		int left = max(0, corner.x);
		int right = min(images[img_idx].cols, corner.x + images[0].cols);
		int up = max(0, corner.y);
		int down = min(images[img_idx].rows, corner.y + images[0].rows);
		for (int y = up; y < down; y++)
			for(int x = left; x < right; x++)
				ec_maps[img_idx](y, x) /= ec_maps[0](y-corner.y, x-corner.x);
	}
	ec_maps[0] = Mat_<float>::ones(ec_maps[0].size());

	ec_maps_ = ec_maps;
}

void MyExposureCompensator::feed( const vector<Point> &corners, const vector<Mat> &images, vector<Mat> &masks )
{
	vector<pair<Mat,uchar> > level_masks;
	for (size_t i = 0; i < masks.size(); ++i)
		level_masks.push_back(make_pair(masks[i], 255));
	createWeightMaps(corners, images, level_masks, ec_maps_);
}

void MyExposureCompensator::gainMapResize( vector<Size> sizes_, vector<Mat_<float>> &ec_maps )
{
	int n = sizes_.size();
	for(int i = 0; i < n; i++)
	{
		Mat_<float> gain_map;
		resize(ec_maps[i], gain_map, sizes_[i], 0, 0, INTER_LINEAR);
		ec_maps[i] = gain_map.clone();
	}
	ec_maps_ = ec_maps;
}

static int total_idx = 0;

void MyExposureCompensator::apply( int index, Mat &image )
{
	CV_Assert(image.type() == CV_8UC3);

	Mat_<float> gain_map;
	if (ec_maps_[index].size() == image.size())
		gain_map = ec_maps_[index];
	else
		resize(ec_maps_[index], gain_map, image.size(), 0, 0, INTER_LINEAR);

	Mat save_map = gain_map * 255;
	save_map.convertTo(save_map, CV_8U);
	char img_name[100];
	sprintf(img_name, "D:/data/my/plane/8-06_jingyuan/ec/debug/ec_%d_%d.jpg", total_idx, index);
	total_idx++;
	imwrite(img_name, save_map);

	for (int y = 0; y < image.rows; ++y)
	{
		const float* gain_row = gain_map.ptr<float>(y);
		Point3_<uchar>* row = image.ptr<Point3_<uchar> >(y);
		for (int x = 0; x < image.cols; ++x)
		{
			row[x].x = saturate_cast<uchar>(row[x].x * gain_row[x]);
			row[x].y = saturate_cast<uchar>(row[x].y * gain_row[x]);
			row[x].z = saturate_cast<uchar>(row[x].z * gain_row[x]);
		}
	}
}
